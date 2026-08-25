// ============================================================================
// 射流轴交换（Axis Switching）— 关键帧动画工作台
// 现象：液体从非圆孔（椭圆/矩形/方孔）喷出后，截面在表面张力与惯性作用下
//       沿流向周期性翻转（椭圆/矩形翻转 90°、方孔翻转 45°），形成链条状水柱。
// 功能：Blender/AE 风格时间轴 + 全参数关键帧 + 摄像机/自由视角切换 + 视频/序列帧导出
// 说明：物理模型移植自原 React 演示（jetPhysics.ts），渲染层重建为工作台场景。
// ============================================================================
import * as THREE from 'three';
import { OrbitControls } from './assets/vendor/OrbitControls.js';

const PREVIEW_FPS = 30;
const NAMES_W = 196;
const LANE_H = 26;
const RULER_H = 30;
const FRAME_SIZE = 46; // 扫描方框边长 mm

// ---------------------------------------------------------------------------
// 0. 物理模型（移植自 jetPhysics.ts）
// ---------------------------------------------------------------------------
const G = 9.81;          // m/s^2
const RHO = 1000;        // kg/m^3
const SIGMA = 0.072;     // N/m
const MU = 1.0e-3;       // Pa·s
const OSC_COEFF = { ellipse: 6, rectangle: 8, square: 12, circle: 0 };
const SUPER_P = 6;
const SHAPES = ['ellipse', 'rectangle', 'circle', 'square'];
const SHAPE_NAMES = ['椭圆', '矩形', '圆孔', '方孔'];

/** 单位等效面积的孔口形状极坐标半径 */
function shapeRadius(shape, k, theta) {
  const c = Math.cos(theta), s = Math.sin(theta);
  switch (shape) {
    case 'circle': return 1;
    case 'ellipse': {
      return 1 / Math.sqrt((c * c) / k + k * s * s);
    }
    case 'rectangle': {
      const hx = Math.sqrt(k), hy = 1 / Math.sqrt(k);
      const cx = Math.abs(c) / hx, sz = Math.abs(s) / hy;
      return Math.pow(Math.pow(cx, SUPER_P) + Math.pow(sz, SUPER_P), -1 / SUPER_P);
    }
    case 'square': {
      const cx = Math.abs(c), sz = Math.abs(s);
      return Math.pow(Math.pow(cx, SUPER_P) + Math.pow(sz, SUPER_P), -1 / SUPER_P);
    }
  }
}

/** 孔口形状旋转后的镜像：椭圆/矩形转 90°，方孔转 45°，圆不变 */
function flippedRadius(shape, k, theta) {
  const rot = shape === 'square' ? Math.PI / 4 : shape === 'circle' ? 0 : Math.PI / 2;
  return shapeRadius(shape, k, theta - rot);
}

/** 形状渐变：w=1 → 出口形状；w=0 → 翻转形状；w=0.5 → 接近圆形 */
function morphRadius(d, theta, w) {
  return w * shapeRadius(d.shape, d.aspectK, theta) + (1 - w) * flippedRadius(d.shape, d.aspectK, theta);
}

const NORM_STEPS = 32;
const normCache = new WeakMap();
function normTable(d) {
  let table = normCache.get(d);
  if (table) return table;
  table = new Float32Array(NORM_STEPS + 1);
  const N = 128;
  for (let wi = 0; wi <= NORM_STEPS; wi++) {
    const w = wi / NORM_STEPS;
    let mean = 0;
    for (let i = 0; i < N; i++) {
      const theta = (i / N) * Math.PI * 2;
      const m = morphRadius(d, theta, w);
      mean += m * m;
    }
    mean /= N;
    table[wi] = 1 / Math.sqrt(mean);
  }
  normCache.set(d, table);
  return table;
}
function normAt(d, w) {
  const table = normTable(d);
  const x = Math.min(Math.max(w, 0), 1) * NORM_STEPS;
  const i = Math.floor(x), f = x - i;
  return table[i] * (1 - f) + table[Math.min(i + 1, NORM_STEPS)] * f;
}

/** 由界面参数推导全部流动/振荡量 */
function deriveJet(p) {
  const d = p.widthMm / 1000;
  const k = Math.max(p.aspect, 1.0001);
  const Q = p.flowMlS * 1e-6;
  let area0;
  switch (p.shape) {
    case 'circle': area0 = (Math.PI * d * d) / 4; break;
    case 'square': area0 = d * d; break;
    case 'ellipse': area0 = Math.PI * (d / 2) * (d / (2 * k)); break;
    case 'rectangle': area0 = d * (d / k); break;
  }
  const v0 = Q / area0;
  const req = Math.sqrt(area0 / Math.PI);
  const weber = (RHO * v0 * v0 * d) / SIGMA;
  const reynolds = (RHO * v0 * d) / MU;
  const C = OSC_COEFF[p.shape];
  const omega = p.shape === 'circle' ? 0 : Math.sqrt((C * SIGMA) / (RHO * req ** 3));
  const firstSwitchZ = omega > 0 ? zAtPhase(v0, Math.PI / omega) : Infinity;
  let jetLength = 0.1;
  if (omega > 0) {
    jetLength = zAtPhase(v0, (7 * Math.PI) / omega) * 1.05;
    jetLength = Math.min(0.22, Math.max(0.07, jetLength));
  }
  const switchAngleDeg = p.shape === 'square' ? 45 : 90;
  return { area0, v0, req, weber, reynolds, omega, firstSwitchZ, jetLength, shape: p.shape, aspectK: k, switchAngleDeg };
}
function zAtPhase(v0, tau) {
  const v = v0 + G * tau;
  return (v * v - v0 * v0) / (2 * G);
}
function velocityAt(v0, z) { return Math.sqrt(v0 * v0 + 2 * G * z); }
function travelTime(v0, z) { return (velocityAt(v0, z) - v0) / G; }
function meanRadiusAt(d, z) { return d.req * Math.sqrt(d.v0 / velocityAt(d.v0, z)); }
function oscillationAt(d, z) {
  if (d.omega === 0) return 0;
  const tau = travelTime(d.v0, z);
  const Q = Math.sqrt(SIGMA * RHO * d.req) / MU;
  const decay = Math.exp((-d.omega * tau) / (Q * 8));
  return Math.cos(d.omega * tau) * decay;
}
function morphWeightAt(d, z) { return 0.5 + 0.5 * oscillationAt(d, z); }
function contourRadiusAt(d, z, theta) {
  const R = meanRadiusAt(d, z);
  if (d.omega === 0) return R;
  const w = morphWeightAt(d, z);
  return R * morphRadius(d, theta, w) * normAt(d, w);
}
function deformationAt(d, z, theta) {
  if (d.omega === 0) return 0;
  const w = morphWeightAt(d, z);
  return morphRadius(d, theta, w) * normAt(d, w) - 1;
}
function computeCrossSection(d, zMeters, samples = 128) {
  const z = Math.min(Math.max(zMeters, 0), d.jetLength);
  const points = [];
  let widthX = 0, widthZ = 0;
  for (let i = 0; i < samples; i++) {
    const theta = (i / samples) * Math.PI * 2;
    const r = contourRadiusAt(d, z, theta);
    const x = r * Math.cos(theta), zz = r * Math.sin(theta);
    points.push([x * 1000, zz * 1000]);
    widthX = Math.max(widthX, Math.abs(x));
    widthZ = Math.max(widthZ, Math.abs(zz));
  }
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    area += x1 * y2 - x2 * y1;
  }
  const tau = travelTime(d.v0, z);
  return {
    zMm: z * 1000, points,
    widthX: widthX * 2 * 1000, widthZ: widthZ * 2 * 1000,
    areaMm2: Math.abs(area) / 2,
    amplitude: oscillationAt(d, z),
    switchCount: d.omega > 0 ? Math.floor((d.omega * tau) / Math.PI) : 0,
    meanRadiusMm: meanRadiusAt(d, z) * 1000,
  };
}
function buildJetSurface(d, nz = 300, nt = 72) {
  const L = d.jetLength;
  const verts = (nz + 1) * (nt + 1);
  const positions = new Float32Array(verts * 3);
  const uvs = new Float32Array(verts * 2);
  const indices = new Uint32Array(nz * nt * 6);
  const V_REPEAT = 10;
  let p = 0, uvi = 0;
  for (let i = 0; i <= nz; i++) {
    const z = (i / nz) * L;
    for (let j = 0; j <= nt; j++) {
      const theta = (j / nt) * Math.PI * 2;
      const r = contourRadiusAt(d, z, theta) * 1000;
      positions[p++] = r * Math.cos(theta);
      positions[p++] = -z * 1000;
      positions[p++] = r * Math.sin(theta);
      uvs[uvi++] = j / nt;
      uvs[uvi++] = (i / nz) * V_REPEAT;
    }
  }
  let q = 0;
  for (let i = 0; i < nz; i++) {
    for (let j = 0; j < nt; j++) {
      const a = i * (nt + 1) + j, b = a + nt + 1;
      indices[q++] = a; indices[q++] = b; indices[q++] = a + 1;
      indices[q++] = b; indices[q++] = b + 1; indices[q++] = a + 1;
    }
  }
  return { positions, uvs, indices, lengthMm: L * 1000 };
}

const DEFAULT_PARAMS = { shape: 'ellipse', widthMm: 3, flowMlS: 8, aspect: 2 };
const L0 = deriveJet(DEFAULT_PARAMS).jetLength * 1000;   // 默认液柱长度 mm
const CAM_DIST = L0 * 1.15 + 55;

// ---------------------------------------------------------------------------
// 1. 可打关键帧的参数轨道定义
// ---------------------------------------------------------------------------
const TRACKS = [
  { g: '摄像机', id: 'camX', label: '位置 X', min: -400, max: 400, step: 1, def: Math.round(CAM_DIST * 0.75) },
  { g: '摄像机', id: 'camY', label: '位置 Y', min: -250, max: 150, step: 1, def: Math.round(-L0 * 0.35) },
  { g: '摄像机', id: 'camZ', label: '位置 Z', min: -400, max: 400, step: 1, def: Math.round(CAM_DIST * 0.75) },
  { g: '摄像机', id: 'rotX', label: '旋转 X°', min: -180, max: 180, step: 1, def: 0 },
  { g: '摄像机', id: 'rotY', label: '旋转 Y°', min: -180, max: 180, step: 1, def: 0 },
  { g: '摄像机', id: 'rotZ', label: '旋转 Z°', min: -180, max: 180, step: 1, def: 0 },
  { g: '摄像机', id: 'fov', label: '焦距 FOV', min: 15, max: 110, step: 0.5, def: 42 },
  { g: '摄像机', id: 'tgtX', label: '视觉中心 X', min: -60, max: 60, step: 0.5, def: 0 },
  { g: '摄像机', id: 'tgtY', label: '视觉中心 Y', min: -160, max: 20, step: 0.5, def: Math.round(-L0 / 2) },
  { g: '摄像机', id: 'tgtZ', label: '视觉中心 Z', min: -60, max: 60, step: 0.5, def: 0 },
  { g: '孔口设置', id: 'shape', label: '孔口形状', min: 0, max: 3, step: 1, def: 0, integer: true, seg: SHAPE_NAMES },
  { g: '孔口设置', id: 'widthMm', label: '开口宽度', min: 0.3, max: 5, step: 0.05, def: 3 },
  { g: '孔口设置', id: 'aspect', label: '长宽比 a/b', min: 1.2, max: 12, step: 0.1, def: 2 },
  { g: '孔口设置', id: 'flowMlS', label: '流量 Q', min: 1, max: 40, step: 0.5, def: 8 },
  { g: '液柱显示', id: 'renderMode', label: '显示模式', min: 0, max: 3, step: 1, def: 0, integer: true, seg: ['玻璃', '实心', '热力图', '线框'] },
  { g: '液柱显示', id: 'flowStripes', label: '流动条纹', min: 0, max: 1, step: 1, def: 1, integer: true, seg: ['关', '开'] },
  { g: '液柱显示', id: 'scanOn', label: '截面扫描', min: 0, max: 1, step: 1, def: 0, integer: true, seg: ['关', '开'] },
  { g: '液柱显示', id: 'scanDepth', label: '扫描深度 z', min: 0, max: 220, step: 0.5, def: 33 },
  { g: '液柱显示', id: 'frontProgress', label: '液柱生长', min: 0, max: 1, step: 0.01, def: 1 },
];
const TRACK_MAP = Object.fromEntries(TRACKS.map(t => [t.id, t]));

// ---------------------------------------------------------------------------
// 2. 全局状态：静态值 + 关键帧表 + 播放状态
// ---------------------------------------------------------------------------
const state = {
  duration: 14,
  time: 0,
  playing: false,
  loop: true,
  px: 95,
  view: 'camera',
  lookAtTarget: true,
  selected: null,
  statics: {},
  keys: {},
};
for (const tr of TRACKS) { state.statics[tr.id] = tr.def; state.keys[tr.id] = []; }

// 预置一段演示动画：液柱生长 + 孔型切换（椭圆→矩形→方孔）+ 相机环绕 + 热力图/截面扫描展示
function seedDemo() {
  const K = (id, arr) => { state.keys[id] = arr.map(([t, v, i]) => ({ t, v, interp: i || 'smooth' })); };
  // 相机环绕
  K('camX', [[0, 122], [3.5, 0], [7, -122], [10.5, 0], [14, 122]]);
  K('camZ', [[0, 0], [3.5, 122], [7, 0], [10.5, -122], [14, 0]]);
  K('camY', [[0, -40], [5, -72], [9, -36], [14, -40]]);
  K('tgtY', [[0, -50], [14, -50]]);
  K('fov', [[0, 42], [7, 38], [14, 42]]);
  // 液柱生长（开孔出水）
  K('frontProgress', [[0, 0], [3, 1], [14, 1]]);
  // 孔型切换：椭圆(90°) → 矩形(90°) → 方孔(45°) → 椭圆
  K('shape', [[0, 0, 'step'], [8, 0, 'step'], [8.1, 1, 'step'], [10.4, 1, 'step'], [10.5, 3, 'step'], [12.4, 3, 'step'], [12.5, 0, 'step'], [14, 0, 'step']]);
  K('widthMm', [[0, 3], [8, 3], [10.5, 2.4], [12.4, 2.4], [12.5, 3], [14, 3]]);
  K('aspect', [[0, 2], [8, 2], [10.5, 3], [12.4, 3], [12.5, 2], [14, 2]]);
  K('flowMlS', [[0, 8], [7, 13], [14, 8]]);
  // 显示模式：中段切到热力图看红蓝振荡带，然后做截面扫描
  K('renderMode', [[0, 0, 'step'], [7.4, 0, 'step'], [7.5, 2, 'step'], [10.3, 2, 'step'], [10.4, 0, 'step'], [14, 0, 'step']]);
  K('flowStripes', [[0, 1, 'step'], [14, 1, 'step']]);
  K('scanOn', [[0, 0, 'step'], [10.7, 0, 'step'], [10.8, 1, 'step'], [13.6, 1, 'step'], [13.7, 0, 'step'], [14, 0, 'step']]);
  K('scanDepth', [[10.8, 8], [13.6, 65]]);
}
seedDemo();

// ---------------------------------------------------------------------------
// 3. Three.js 场景
// ---------------------------------------------------------------------------
const viewport = document.getElementById('viewport');
const viewportWrap = document.getElementById('viewport-wrap');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: true });
renderer.setPixelRatio(1);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.localClippingEnabled = true;
renderer.setSize(viewport.clientWidth || 800, viewport.clientHeight || 600, false);
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1526);

// 简易环境贴图（玻璃质感需要环境反射）
{
  const pmrem = new THREE.PMREMGenerator(renderer);
  const es = new THREE.Scene();
  const panel = (color, x, y, z, w, h, d) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshBasicMaterial({ color }));
    m.position.set(x, y, z);
    es.add(m);
  };
  panel(0xffffff, 0, 8, -9, 14, 10, 0.6);
  panel(0xfff1d6, 10, 5, 6, 8, 5, 0.5);
  panel(0xbfe0ff, -11, 3, -3, 8, 5, 0.5);
  panel(0xd9e8ff, 0, -4, 10, 12, 6, 0.5);
  scene.environment = pmrem.fromScene(es, 0.05).texture;
  pmrem.dispose();
}

const camera = new THREE.PerspectiveCamera(42, 1, 0.5, 5000);
camera.position.set(CAM_DIST * 0.75, -L0 * 0.35, CAM_DIST * 0.75);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 10;
controls.maxDistance = 1000;
controls.target.set(0, -L0 / 2, 0);
controls.enabled = false;

// 灯光
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(80, 60, 60);
scene.add(key);
const rim = new THREE.DirectionalLight(0x7fb4ff, 1.2);
rim.position.set(-70, -40, -60);
scene.add(rim);
scene.add(new THREE.AmbientLight(0x8899bb, 0.5));

// 参考地面网格
const grid = new THREE.GridHelper(300, 30, 0x1d2f52, 0x14233f);
scene.add(grid);

// 裁剪平面：液柱前端从孔口缓缓下落（播放"开孔出水"）
const clipPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 1e9);

// --- 液柱刚性组件：表面 / 流动条纹 / 流线笼 / 孔板 / 扫描平面 / 前端高亮 ---
const jet = (() => {
  let jetMesh = null, jetMat = null, overlay = null, cage = null, cageMat = null;
  let solidMat = null, heatMat = null, wireMat = null;
  let plateGroup = null, plateMat = null;
  let stripeTex = null;
  let derived = null;
  let mode = 0;          // 0玻璃 1实心 2热力图 3线框
  let dim = false;       // 截面扫描时调暗液柱
  let stripesOn = true;

  // 扫描平面组（白框 + 半透明面 + 高亮截面）
  const scanGroup = new THREE.Group();
  {
    const t = 1.1;
    const mat = new THREE.MeshBasicMaterial({ color: 0xf1f5f9 });
    const mk = (w, h, x, z) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, t, h), mat);
      m.position.set(x, 0, z);
      scanGroup.add(m);
    };
    const S = FRAME_SIZE;
    mk(S, t, 0, -S / 2); mk(S, t, 0, S / 2);
    mk(t, S, -S / 2, 0); mk(t, S, S / 2, 0);
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(S - t, S - t),
      new THREE.MeshBasicMaterial({ color: 0x9cc8e8, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false, toneMapped: false }),
    );
    plane.rotation.x = -Math.PI / 2;
    plane.renderOrder = 18;
    scanGroup.add(plane);
    scanGroup.visible = false;
    scene.add(scanGroup);
  }
  let sectionMesh = null, sectionHalo = null, sectionLine = null;
  let frontFill = null, frontLine = null;

  // 扫描平面文字标签
  const scanLabel = document.createElement('div');
  scanLabel.className = 'scan-label';
  scanLabel.textContent = '';
  viewportWrap.appendChild(scanLabel);

  function stripeTexture() {
    if (stripeTex) return stripeTex;
    const c = document.createElement('canvas');
    c.width = 4; c.height = 128;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, 4, 128);
    const g = ctx.createLinearGradient(0, 0, 0, 128);
    g.addColorStop(0, 'rgba(170,230,255,0)');
    g.addColorStop(0.38, 'rgba(190,238,255,0.85)');
    g.addColorStop(0.5, 'rgba(255,255,255,1)');
    g.addColorStop(0.62, 'rgba(190,238,255,0.85)');
    g.addColorStop(1, 'rgba(170,230,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 4, 128);
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    stripeTex = tex;
    return tex;
  }

  function buildPlate() {
    if (plateGroup) {
      scene.remove(plateGroup);
      plateGroup.traverse(o => {
        if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); }
      });
    }
    const p = (derived && derived.__params) ? derived.__params : DEFAULT_PARAMS;
    const d = p.widthMm;
    const k = Math.max(p.aspect, 1.0001);
    const size = Math.max(14, d + 7);
    const shape = new THREE.Shape();
    shape.moveTo(-size / 2, -size / 2);
    shape.lineTo(size / 2, -size / 2);
    shape.lineTo(size / 2, size / 2);
    shape.lineTo(-size / 2, size / 2);
    shape.closePath();
    const hole = new THREE.Path();
    const shapeId = p.shape;
    if (shapeId === 'circle') hole.absarc(0, 0, d / 2, 0, Math.PI * 2, true);
    else if (shapeId === 'ellipse') hole.absellipse(0, 0, d / 2, d / (2 * k), 0, Math.PI * 2, true);
    else {
      const hx = d / 2;
      const hz = shapeId === 'square' ? d / 2 : d / (2 * k);
      hole.moveTo(-hx, -hz); hole.lineTo(hx, -hz); hole.lineTo(hx, hz); hole.lineTo(-hx, hz); hole.closePath();
    }
    shape.holes.push(hole);
    const geo = new THREE.ExtrudeGeometry(shape, { depth: 0.6, bevelEnabled: false });
    const mat = new THREE.MeshStandardMaterial({
      color: 0x334155, metalness: 0.3, roughness: 0.7, transparent: true, opacity: 0.75, depthWrite: false,
    });
    plateMat = mat;
    const plate = new THREE.Mesh(geo, mat);
    plate.rotation.x = Math.PI / 2;
    plate.position.y = 0.3;
    plateGroup = new THREE.Group();
    plateGroup.add(plate);
    scene.add(plateGroup);
  }

  function rebuild(d, params) {
    derived = d;
    if (params) derived.__params = params;

    const { positions, uvs, indices } = buildJetSurface(d);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));

    // 热力图顶点色：按带符号径向变形着色
    const vertCount = positions.length / 3;
    const colors = new Float32Array(vertCount * 3);
    const nz = 300, nt = 72;
    const pale = [0.87, 0.95, 0.99], warm = [0.95, 0.42, 0.09], cool = [0.23, 0.45, 0.95];
    let ci = 0;
    for (let i = 0; i <= nz; i++) {
      const z = (i / nz) * d.jetLength;
      for (let j = 0; j <= nt; j++) {
        const theta = (j / nt) * Math.PI * 2;
        const t = Math.max(-1, Math.min(1, deformationAt(d, z, theta) / 0.2));
        const tgt = t >= 0 ? warm : cool;
        const w = Math.abs(t);
        colors[ci++] = pale[0] + (tgt[0] - pale[0]) * w;
        colors[ci++] = pale[1] + (tgt[1] - pale[1]) * w;
        colors[ci++] = pale[2] + (tgt[2] - pale[2]) * w;
      }
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();

    if (!jetMesh) {
      const mat = new THREE.MeshPhysicalMaterial({
        color: 0x9fd8f0, transmission: 0.85, thickness: 3, roughness: 0.08, ior: 1.33,
        transparent: true, opacity: 0.92, side: THREE.DoubleSide, clearcoat: 0.6, clearcoatRoughness: 0.2,
      });
      mat.clippingPlanes = [clipPlane];
      jetMat = mat;
      jetMesh = new THREE.Mesh(geo, mat);
      scene.add(jetMesh);

      const om = new THREE.MeshBasicMaterial({
        map: stripeTexture(), transparent: true, opacity: 0.28,
        blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
      });
      om.clippingPlanes = [clipPlane];
      overlay = new THREE.Mesh(geo, om);
      overlay.renderOrder = 2;
      scene.add(overlay);

      const cm = new THREE.LineBasicMaterial({ color: 0x155e75, transparent: true, opacity: 0.45 });
      cm.clippingPlanes = [clipPlane];
      cageMat = cm;
      cage = new THREE.LineSegments(new THREE.BufferGeometry(), cm);
      scene.add(cage);
    } else {
      jetMesh.geometry.dispose();
      jetMesh.geometry = geo;
      overlay.geometry = geo;
    }

    // 纵向"流线笼"：表面固定方位角细线 → 轴交换扭转清晰可见
    const NL = 24, NZ = 160;
    const cagePos = new Float32Array(NL * (NZ - 1) * 2 * 3);
    let cp = 0;
    for (let j = 0; j < NL; j++) {
      const theta = (j / NL) * Math.PI * 2;
      let px = 0, py = 0, pz = 0;
      for (let i = 0; i < NZ; i++) {
        const z = (i / (NZ - 1)) * d.jetLength;
        const r = contourRadiusAt(d, z, theta) * 1000 + 0.08;
        const x = r * Math.cos(theta), y = -z * 1000, zz = r * Math.sin(theta);
        if (i > 0) {
          cagePos[cp++] = px; cagePos[cp++] = py; cagePos[cp++] = pz;
          cagePos[cp++] = x; cagePos[cp++] = y; cagePos[cp++] = zz;
        }
        px = x; py = y; pz = zz;
      }
    }
    cage.geometry.dispose();
    cage.geometry.setAttribute('position', new THREE.BufferAttribute(cagePos, 3));

    buildPlate();
    grid.position.y = -d.jetLength * 1000 - 14;
    applyDim();
  }

  function setMode(m) {
    mode = m;
    if (!jetMesh) return;
    if (m === 0) {
      jetMesh.material = jetMat;
    } else {
      if (m === 1 && !solidMat) {
        solidMat = new THREE.MeshStandardMaterial({ color: 0x6ec6e8, roughness: 0.3, metalness: 0.05, side: THREE.DoubleSide });
        solidMat.clippingPlanes = [clipPlane];
      }
      if (m === 2 && !heatMat) {
        heatMat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
        heatMat.clippingPlanes = [clipPlane];
      }
      if (m === 3 && !wireMat) {
        wireMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true });
        wireMat.clippingPlanes = [clipPlane];
      }
      jetMesh.material = m === 1 ? solidMat : m === 2 ? heatMat : wireMat;
    }
    if (cage) cage.visible = m !== 3;
    if (cageMat) cageMat.opacity = m === 0 ? 0.45 : 0.75;
    if (overlay) overlay.visible = stripesOn && m !== 3;
    applyDim();
  }

  function setStripes(on) {
    stripesOn = on;
    if (overlay) overlay.visible = on && mode !== 3;
  }

  function setScan(on) {
    dim = on;
    scanGroup.visible = on;
    scanLabel.style.display = on ? 'block' : 'none';
    applyDim();
  }

  function applyDim() {
    if (jetMat) {
      jetMat.transmission = dim ? 0 : 0.85;
      jetMat.opacity = dim ? 0.28 : 0.92;
      jetMat.depthWrite = !dim;
      jetMat.needsUpdate = true;
    }
    for (const m of [solidMat, heatMat, wireMat]) {
      if (!m) continue;
      const nextTransparent = dim || m.wireframe;
      if (m.transparent !== nextTransparent) m.transparent = nextTransparent;
      m.opacity = dim ? 0.3 : 1;
      m.depthWrite = !dim;
      m.needsUpdate = true;
    }
    if (cageMat) cageMat.opacity = dim ? 0.15 : (mode === 0 ? 0.45 : 0.75);
  }

  function shapeFromSection(section) {
    const shape = new THREE.Shape();
    section.points.forEach(([x, z], i) => { if (i === 0) shape.moveTo(x, z); else shape.lineTo(x, z); });
    shape.closePath();
    return shape;
  }

  function updateScan(zMm) {
    scanGroup.position.y = -zMm;
    const section = computeCrossSection(derived, zMm / 1000);

    if (sectionMesh) { sectionMesh.geometry.dispose(); scanGroup.remove(sectionMesh); }
    const fillGeo = new THREE.ShapeGeometry(shapeFromSection(section));
    fillGeo.rotateX(-Math.PI / 2);
    sectionMesh = new THREE.Mesh(fillGeo, new THREE.MeshBasicMaterial({
      color: 0x4df3ff, transparent: true, opacity: 1.0, side: THREE.DoubleSide, depthWrite: false, toneMapped: false,
    }));
    sectionMesh.position.y = 0.06;
    sectionMesh.renderOrder = 20;
    scanGroup.add(sectionMesh);

    if (sectionHalo) { sectionHalo.geometry.dispose(); scanGroup.remove(sectionHalo); }
    const haloShape = new THREE.Shape();
    section.points.forEach(([x, z], i) => {
      const hx = x * 1.28, hz = z * 1.28;
      if (i === 0) haloShape.moveTo(hx, hz); else haloShape.lineTo(hx, hz);
    });
    haloShape.closePath();
    const haloGeo = new THREE.ShapeGeometry(haloShape);
    haloGeo.rotateX(-Math.PI / 2);
    sectionHalo = new THREE.Mesh(haloGeo, new THREE.MeshBasicMaterial({
      color: 0x22d3ee, transparent: true, opacity: 0.38, side: THREE.DoubleSide, depthWrite: false, toneMapped: false,
    }));
    sectionHalo.position.y = 0.03;
    sectionHalo.renderOrder = 19;
    scanGroup.add(sectionHalo);

    if (sectionLine) { sectionLine.geometry.dispose(); scanGroup.remove(sectionLine); }
    const linePts = section.points.map(([x, z]) => new THREE.Vector3(x, 0.14, z));
    sectionLine = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(linePts),
      new THREE.LineBasicMaterial({ color: 0xecfeff, toneMapped: false }));
    sectionLine.renderOrder = 21;
    scanGroup.add(sectionLine);

    // 标签跟随投影
    const v = new THREE.Vector3(FRAME_SIZE / 2 + 3, -zMm, 0).project(camera);
    const w = viewportWrap.clientWidth, h = viewportWrap.clientHeight;
    scanLabel.style.left = (((v.x + 1) / 2) * w) + 'px';
    scanLabel.style.top = (((-v.y + 1) / 2) * h - 8) + 'px';
    scanLabel.textContent = `扫描平面  z = ${zMm.toFixed(1)} mm`;
  }

  function updateFront(frontMm) {
    const section = computeCrossSection(derived, frontMm / 1000, 72);
    const fillGeo = new THREE.ShapeGeometry(shapeFromSection(section));
    fillGeo.rotateX(-Math.PI / 2);
    if (!frontFill) {
      frontFill = new THREE.Mesh(fillGeo, new THREE.MeshBasicMaterial({
        color: 0xffb020, transparent: true, opacity: 0.95, side: THREE.DoubleSide, depthWrite: false, toneMapped: false,
      }));
      frontFill.renderOrder = 15;
      scene.add(frontFill);
    } else { frontFill.geometry.dispose(); frontFill.geometry = fillGeo; }
    frontFill.position.y = -frontMm;

    const pts = section.points.map(([x, z]) => new THREE.Vector3(x, 0.05, z));
    const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
    if (!frontLine) {
      frontLine = new THREE.LineLoop(lineGeo, new THREE.LineBasicMaterial({ color: 0xfff3d0, toneMapped: false }));
      frontLine.renderOrder = 16;
      scene.add(frontLine);
    } else { frontLine.geometry.dispose(); frontLine.geometry = lineGeo; }
    frontLine.position.y = -frontMm;
  }

  function showFront(show) {
    if (frontFill) frontFill.visible = show;
    if (frontLine) frontLine.visible = show;
  }

  return {
    get derived() { return derived; },
    rebuild,
    setMode,
    setStripes,
    setScan,
    updateScan,
    updateFront,
    showFront,
    get stripeTex() { return stripeTex; },
    get plateMat() { return plateMat; },
  };
})();

// 初始构建（默认参数）
  jet.rebuild(deriveJet(DEFAULT_PARAMS), DEFAULT_PARAMS);

// ---------------------------------------------------------------------------
// 4. 关键帧求值（Catmull-Rom 平滑 / 线性 / 阶梯）
// ---------------------------------------------------------------------------
function keysOf(id) { return state.keys[id]; }
function evalTrack(id, t) {
  const ks = keysOf(id);
  if (!ks.length) return state.statics[id];
  if (t <= ks[0].t) return ks[0].v;
  if (t >= ks[ks.length - 1].t) return ks[ks.length - 1].v;
  let i = 0;
  while (i < ks.length - 2 && ks[i + 1].t <= t) i++;
  const k0 = ks[Math.max(0, i - 1)], k1 = ks[i], k2 = ks[i + 1], k3 = ks[Math.min(ks.length - 1, i + 2)];
  const span = k2.t - k1.t;
  const u = span > 1e-9 ? (t - k1.t) / span : 0;
  if (k1.interp === 'step') return k1.v;
  if (k1.interp === 'linear') return k1.v + (k2.v - k1.v) * u;
  const t1 = u, t2 = t1 * u, t3 = t2 * u;
  return 0.5 * ((2 * k1.v) + (-k0.v + k2.v) * t1 +
    (2 * k0.v - 5 * k1.v + 4 * k2.v - k3.v) * t2 +
    (-k0.v + 3 * k1.v - 3 * k2.v + k3.v) * t3);
}
function currentValue(id) {
  const v = evalTrack(id, state.time);
  return TRACK_MAP[id].integer ? Math.round(v) : v;
}
function keyIndexAt(id, t, tol = 0.5 / PREVIEW_FPS) {
  return keysOf(id).findIndex(k => Math.abs(k.t - t) <= tol);
}
function upsertKey(id, t, v, interp) {
  const ks = keysOf(id);
  const idx = keyIndexAt(id, t);
  if (idx >= 0) { ks[idx].v = v; if (interp) ks[idx].interp = interp; }
  else {
    ks.push({ t, v, interp: interp || (TRACK_MAP[id].seg ? 'step' : 'smooth') });
    ks.sort((a, b) => a.t - b.t);
  }
}
function removeKey(id, index) {
  if (index < 0) return;
  state.keys[id].splice(index, 1);
  if (state.selected && state.selected.trackId === id) state.selected = null;
}

// ---------------------------------------------------------------------------
// 4a. 撤销 / 重做（快照式，作用于全部关键帧轨道 state.keys）
//     每次破坏性修改前 snapshot() 一次；同一手势（拖动数值/菱形/滑杆）内
//     自动合并为一步，避免一次拖动产生几十个撤销节点。
// ---------------------------------------------------------------------------
const MAX_UNDO = 50;
const undoStack = [];
const redoStack = [];
let gestureId = 0;
let snapGesture = -1;
let lastSnapAt = 0;
function beginGesture() { gestureId++; }
function cloneKeys() {
  const out = {};
  for (const id in state.keys) out[id] = state.keys[id].map(k => ({ ...k }));
  return out;
}
function snapshot() {
  const now = Date.now();
  if (snapGesture === gestureId && now - lastSnapAt < 500) return; // 同手势连续变更合并
  snapGesture = gestureId; lastSnapAt = now;
  undoStack.push(cloneKeys());
  if (undoStack.length > MAX_UNDO) undoStack.shift();
  redoStack.length = 0;
  updateUndoButtons();
}
function restoreKeys(snap) {
  const out = {};
  for (const id in state.keys) out[id] = (snap[id] || []).map(k => ({ ...k }));
  return out;
}
function afterKeysChanged() {
  if (state.selected) {
    const k = keysOf(state.selected.trackId)[state.selected.index];
    if (!k) state.selected = null;
  }
  closeKfEditor();
  renderTimeline();
  applyAll(state.time);
  updateUndoButtons();
}
function undo() {
  if (!undoStack.length) { flashHint('没有可撤回的操作'); return; }
  redoStack.push(cloneKeys());
  state.keys = restoreKeys(undoStack.pop());
  snapGesture = -1; lastSnapAt = 0;
  afterKeysChanged();
  flashHint('↩ 已撤回');
}
function redo() {
  if (!redoStack.length) { flashHint('没有可重做的操作'); return; }
  undoStack.push(cloneKeys());
  state.keys = restoreKeys(redoStack.pop());
  snapGesture = -1; lastSnapAt = 0;
  afterKeysChanged();
  flashHint('↪ 已重做');
}
function updateUndoButtons() {
  const u = document.getElementById('btn-undo');
  const r = document.getElementById('btn-redo');
  if (u) u.disabled = !undoStack.length;
  if (r) r.disabled = !redoStack.length;
}

// 修改参数值 → 在当前播放头时间自动创建/更新关键帧
function commitValue(tr, raw) {
  snapshot();
  let val = parseFloat(raw);
  if (isNaN(val)) return;
  val = Math.min(tr.max, Math.max(tr.min, val));
  if (tr.integer) val = Math.round(val);
  upsertKey(tr.id, state.time, val);
  renderTimeline();
  applyAll(state.time);
}

// 参数数值拖拽微调
function makeScrub(el, tr) {
  let scrub = null;
  el.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    beginGesture();
    scrub = { startX: e.clientX, startVal: currentValue(tr.id), moved: false, shift: e.shiftKey };
    el.classList.add('scrubbing');
    try { el.setPointerCapture(e.pointerId); } catch (_) {}
  });
  el.addEventListener('pointermove', e => {
    if (!scrub) return;
    const dx = e.clientX - scrub.startX;
    if (Math.abs(dx) > 3) scrub.moved = true;
    const base = (tr.max - tr.min) / 200;
    const step = scrub.shift ? base * 0.1 : base;
    let val = scrub.startVal + dx * step;
    val = Math.min(tr.max, Math.max(tr.min, val));
    if (tr.integer) val = Math.round(val);
    commitValue(tr, val);
  });
  const end = () => {
    if (!scrub) return;
    const wasMoved = scrub.moved;
    scrub = null;
    el.classList.remove('scrubbing');
    if (!wasMoved && el.tagName === 'INPUT') el.focus();
  };
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
}

// ---------------------------------------------------------------------------
// 5. 把当前时间的求值结果应用到场景 + 面板
// ---------------------------------------------------------------------------
let panelSyncLock = false;
let lastPhys = { shape: -1, width: -1, flow: -1, aspect: -1 };
let lastFront = -1;
let lastScanDepth = -1;
let lastScanOn = -1;
let lastMode = -1;
let lastStripes = -1;

function ensureDerived(v) {
  if (lastPhys.shape !== v.shape ||
      Math.abs(lastPhys.width - v.widthMm) > 1e-9 ||
      Math.abs(lastPhys.flow - v.flowMlS) > 1e-9 ||
      Math.abs(lastPhys.aspect - v.aspect) > 1e-9) {
    lastPhys = { shape: v.shape, width: v.widthMm, flow: v.flowMlS, aspect: v.aspect };
    const params = { shape: SHAPES[v.shape], widthMm: v.widthMm, flowMlS: v.flowMlS, aspect: v.aspect };
    jet.rebuild(deriveJet(params), params);
    lastFront = -1; lastScanDepth = -1;
  }
}

function applyAll(t, forceCamera = false) {
  const v = {};
  for (const tr of TRACKS) v[tr.id] = tr.integer ? Math.round(evalTrack(tr.id, t)) : evalTrack(tr.id, t);

  // --- 物理参数 → 重建液柱 ---
  ensureDerived(v);
  const L = jet.derived.jetLength * 1000;

  // --- 显示模式 / 流动条纹 / 扫描 ---
  if (v.renderMode !== lastMode) { lastMode = v.renderMode; jet.setMode(v.renderMode); }
  if (v.flowStripes !== lastStripes) { lastStripes = v.flowStripes; jet.setStripes(v.flowStripes === 1); }
  if (v.scanOn !== lastScanOn) { lastScanOn = v.scanOn; jet.setScan(v.scanOn === 1); }
  if (jet.derived && v.scanOn === 1) {
    const effDepth = Math.min(v.scanDepth, Math.max(2, L - 2));
    if (Math.abs(effDepth - lastScanDepth) > 1e-6) { lastScanDepth = effDepth; jet.updateScan(effDepth); }
  }

  // --- 液柱生长（裁剪平面 + 前端高亮） ---
  const front = v.frontProgress * L;
  if (v.frontProgress >= 0.999) {
    if (lastFront !== 1) { lastFront = 1; clipPlane.constant = 1e9; jet.showFront(false); }
  } else {
    clipPlane.constant = front;
    if (Math.abs(front - lastFront) > 0.01) { lastFront = front; jet.updateFront(front); }
    jet.showFront(true);
  }

  // --- 流动条纹向下滚动（随平均流速） ---
  if (jet.stripeTex && v.flowStripes === 1 && v.renderMode !== 3) {
    const meanV = (jet.derived.v0 + Math.sqrt(jet.derived.v0 ** 2 + 2 * 9.81 * jet.derived.jetLength)) / 2;
    const rate = (meanV * 1000 * 0.015) / L;
    jet.stripeTex.offset.y = -(t * rate) % 1;
  }

  // 摄像机
  const driveCamera = forceCamera || state.view === 'camera';
  if (driveCamera) {
    camera.position.set(v.camX, v.camY, v.camZ);
    if (state.lookAtTarget) {
      camera.lookAt(v.tgtX, v.tgtY, v.tgtZ);
    } else {
      camera.rotation.set(THREE.MathUtils.degToRad(v.rotX), THREE.MathUtils.degToRad(v.rotY), THREE.MathUtils.degToRad(v.rotZ), 'YXZ');
    }
    if (camera.fov !== v.fov) { camera.fov = v.fov; camera.updateProjectionMatrix(); }
  }
  // 自由视角下允许用户平移视觉中心（右键拖拽），不强制写回关键帧求值
  if (driveCamera) controls.target.set(v.tgtX, v.tgtY, v.tgtZ);

  // 从上方俯视时让孔板近乎隐形
  const pmat = jet.plateMat;
  if (pmat) {
    const dir = camera.position.clone().sub(controls.target);
    const len = dir.length();
    if (len > 1e-6) {
      const polar = Math.acos(Math.min(1, Math.max(-1, dir.y / len)));
      const tFade = Math.min(Math.max((polar - 0.35) / 0.5, 0), 1);
      pmat.opacity = 0.08 + tFade * 0.67;
    }
  }

  syncPanel(v);
  updateTimeDisplay();
  updatePlayhead();
  updateKfButtons();
}

// ---------------------------------------------------------------------------
// 6. 右侧参数面板
// ---------------------------------------------------------------------------
const panel = document.getElementById('panel');
const panelInputs = {}; // id -> {range, num, kfBtn, segBtns?}
{
  let lastGroup = null;
  for (const tr of TRACKS) {
    if (tr.g !== lastGroup) {
      const h = document.createElement('h3'); h.textContent = tr.g; panel.appendChild(h);
      if (tr.g === '摄像机') {
        const chk = document.createElement('label');
        chk.className = 'chkrow';
        chk.innerHTML = `<input type="checkbox" id="chk-lookat" checked/> 摄像机始终看向视觉中心（取消后用旋转关键帧控制朝向）`;
        panel.appendChild(chk);
        chk.querySelector('input').addEventListener('change', e => { state.lookAtTarget = e.target.checked; applyAll(state.time); });
      }
      lastGroup = tr.g;
    }
    const row = document.createElement('div');
    row.className = 'prow' + (tr.seg ? ' seg' : '');
    const kfBtn = `<button class="kfbtn" data-kf="${tr.id}" title="在当前时间添加/移除关键帧">◇</button>`;
    if (tr.seg) {
      row.innerHTML = `<div class="pname" title="${tr.label}">${tr.label}</div>
        <div class="segbtns">${tr.seg.map((s, i) => `<button data-seg="${tr.id}" data-v="${i}">${s}</button>`).join('')}</div>${kfBtn}`;
    } else {
      row.innerHTML = `<div class="pname" title="${tr.label}">${tr.label}</div>
        <input type="range" data-id="${tr.id}" min="${tr.min}" max="${tr.max}" step="${tr.step}"/>
        <input type="number" data-id="${tr.id}" min="${tr.min}" max="${tr.max}" step="${tr.step}"/>${kfBtn}`;
    }
    panel.appendChild(row);
  }
  for (const tr of TRACKS) {
    panelInputs[tr.id] = {
      range: panel.querySelector(`input[type=range][data-id="${tr.id}"]`),
      num: panel.querySelector(`input[type=number][data-id="${tr.id}"]`),
      kfBtn: panel.querySelector(`button[data-kf="${tr.id}"]`),
      segs: tr.seg ? [...panel.querySelectorAll(`button[data-seg="${tr.id}"]`)] : null,
    };
  }
  for (const tr of TRACKS) {
    const pi = panelInputs[tr.id];
    const commit = raw => commitValue(tr, raw);
    if (pi.range) {
      pi.range.addEventListener('pointerdown', beginGesture);
      pi.range.addEventListener('input', e => commit(e.target.value));
      pi.num.addEventListener('change', e => commit(e.target.value));
      makeScrub(pi.num, tr);
    }
    if (pi.segs) pi.segs.forEach(b => b.addEventListener('click', () => commit(b.dataset.v)));
    pi.kfBtn.addEventListener('click', () => {
      snapshot();
      const idx = keyIndexAt(tr.id, state.time);
      if (idx >= 0) removeKey(tr.id, idx);
      else upsertKey(tr.id, state.time, currentValue(tr.id));
      renderTimeline(); applyAll(state.time);
    });
  }
}
function syncPanel(v) {
  if (panelSyncLock) return;
  for (const tr of TRACKS) {
    const pi = panelInputs[tr.id];
    const val = v[tr.id];
    if (pi.range && document.activeElement !== pi.range) pi.range.value = val;
    if (pi.num && document.activeElement !== pi.num) pi.num.value = tr.integer ? val : (+val).toFixed(2);
    if (pi.segs) pi.segs.forEach(b => b.classList.toggle('on', +b.dataset.v === Math.round(val)));
    const tv = document.querySelector(`.tl-name .tval[data-id="${tr.id}"]`);
    if (tv) tv.textContent = tr.integer ? val : (+val).toFixed(2);
  }
}
function updateKfButtons() {
  for (const tr of TRACKS) {
    panelInputs[tr.id].kfBtn.classList.toggle('active', keyIndexAt(tr.id, state.time) >= 0);
    const laneBtn = document.querySelector(`.tl-name .kfbtn[data-kf="${tr.id}"]`);
    if (laneBtn) laneBtn.classList.toggle('active', keyIndexAt(tr.id, state.time) >= 0);
  }
}

// ---------------------------------------------------------------------------
// 7. 音频口播（外部音频导入 · 波形对齐关键帧节奏）
// ---------------------------------------------------------------------------
const AUDIO_ROW_H = 58;
const audioState = {
  el: null, url: null, name: '', duration: 0,
  peaks: null, ready: false,
  wave: null, waveCtx: null, mask: null, wrap: null
};
const btnAudio = document.getElementById('btn-audio');
const fileAudio = document.getElementById('file-audio');
const audioChip = document.getElementById('audio-chip');
const hintEl = document.querySelector('#topbar .hint');
let hintTimer = null;
function flashHint(msg) {
  if (!hintEl) return;
  hintEl.textContent = msg; hintEl.style.color = '#8fd0ff';
  clearTimeout(hintTimer);
  hintTimer = setTimeout(() => {
    hintEl.textContent = '空格 播放/暂停 · ←/→ 逐帧 · Delete 删除所选关键帧 · 双击轨道空白处添加关键帧 · 拖动数值改参数（自动打帧） · 标尺/轨道拖动跳转 · 🎙 导入口播对齐节奏 · ⌘Z/⌃Z 撤回 · ⌘⇧Z/⌃⇧Z 重做';
    hintEl.style.color = '';
  }, 7000);
}

async function importAudio(file) {
  try {
    const buf = await file.arrayBuffer();
    const AC = window.AudioContext || window.webkitAudioContext;
    const ac = new AC();
    const decoded = await ac.decodeAudioData(buf);
    ac.close();

    const buckets = 2000;
    const ch = decoded.getChannelData(0);
    const per = Math.max(1, Math.floor(ch.length / buckets));
    const peaks = new Float32Array(buckets);
    for (let i = 0; i < buckets; i++) {
      let m = 0;
      const start = i * per;
      for (let j = 0; j < per; j++) {
        const a = Math.abs(ch[start + j] || 0);
        if (a > m) m = a;
      }
      peaks[i] = m;
    }

    if (audioState.el) { audioState.el.pause(); audioState.el.src = ''; }
    if (audioState.url) URL.revokeObjectURL(audioState.url);
    const url = URL.createObjectURL(file);
    const el = new Audio();
    el.preload = 'auto';
    el.src = url;
    Object.assign(audioState, { el, url, name: file.name, duration: decoded.duration, peaks, ready: true });
    el.addEventListener('ended', () => { if (state.playing) setPlaying(false); });

    const need = Math.max(state.duration, Math.ceil(decoded.duration));
    if (need !== state.duration) {
      snapshot();
      state.duration = need;
      document.getElementById('inp-duration').value = need;
      for (const tr of TRACKS) keysOf(tr.id).forEach(k => { k.t = Math.min(k.t, state.duration); });
    }

    audioChip.style.display = 'inline-flex';
    document.getElementById('audio-chip-name').textContent = file.name;
    document.getElementById('audio-chip-dur').textContent = decoded.duration.toFixed(1) + 's';
    document.getElementById('exp-mix-row').style.display = 'flex';
    renderTimeline();
    syncAudioTime();
    flashHint('已导入口播：「' + file.name + '」 ' + decoded.duration.toFixed(1) + 's — 波形已显示在时间轴，播放同步、点击波形可跳转。动画时长已调整为 ' + need + 's');
  } catch (err) {
    alert('音频解码失败：' + err.message);
  }
}

function removeAudio() {
  if (audioState.el) { audioState.el.pause(); audioState.el.src = ''; }
  if (audioState.url) URL.revokeObjectURL(audioState.url);
  audioState.ready = false; audioState.peaks = null;
  audioState.el = null; audioState.url = null; audioState.name = ''; audioState.duration = 0;
  audioChip.style.display = 'none';
  document.getElementById('exp-mix-row').style.display = 'none';
  renderTimeline();
  flashHint('已移除口播音频');
}

function syncAudioTime() {
  if (audioState.ready && audioState.el && !state.exportLive) {
    const t = Math.min(audioState.duration, state.time);
    if (Math.abs(audioState.el.currentTime - t) > 0.06) audioState.el.currentTime = t;
  }
  if (audioState.mask) audioState.mask.style.width = (state.time * state.px) + 'px';
}

btnAudio.addEventListener('click', () => fileAudio.click());
fileAudio.addEventListener('change', () => {
  const f = fileAudio.files[0];
  if (f) importAudio(f);
  fileAudio.value = '';
});
audioChip.addEventListener('click', removeAudio);

// ---------------------------------------------------------------------------
// 8. 时间轴 UI
// ---------------------------------------------------------------------------
const tlBody = document.getElementById('tl-body');
const tlContent = document.getElementById('tl-content');
const playheadEl = document.getElementById('playhead');
const laneEls = {}; // id -> lane div
let rulerCanvas, rulerCtx;

function trackWidth() { return state.duration * state.px; }

function buildTimeline() {
  [...tlContent.querySelectorAll('.tl-row')].forEach(el => el.remove());

  const rulerRow = document.createElement('div');
  rulerRow.className = 'tl-row ruler-row';
  rulerRow.style.height = RULER_H + 'px';
  rulerRow.innerHTML = `<div class="tl-corner">时间（秒）</div>`;
  rulerCanvas = document.createElement('canvas');
  rulerCanvas.id = 'ruler';
  rulerCanvas.height = RULER_H;
  rulerRow.appendChild(rulerCanvas);
  tlContent.appendChild(rulerRow);

  const audioRow = document.createElement('div');
  audioRow.className = 'tl-row audio-row';
  audioRow.style.height = AUDIO_ROW_H + 'px';
  audioRow.innerHTML = `
    <div class="tl-name">
      <span class="tlabel">${audioState.ready ? '🎙 ' + audioState.name : '🎙 音频口播'}</span>
      ${audioState.ready ? `<span class="audio-dur">${audioState.duration.toFixed(2)}s</span>` : ''}
    </div>
    <div id="audio-wave-wrap">
      <canvas id="audio-wave"></canvas>
      <div id="audio-wave-mask"></div>
      <div id="audio-wave-empty" ${audioState.ready ? 'style="display:none"' : ''}>🎙 <span>导入口播音频 — 波形显示于此，播放同步、点击跳转、对齐关键帧节奏</span></div>
    </div>`;
  tlContent.appendChild(audioRow);
  audioState.wave = audioRow.querySelector('#audio-wave');
  audioState.waveCtx = audioState.wave.getContext('2d');
  audioState.mask = audioRow.querySelector('#audio-wave-mask');
  audioState.wrap = audioRow.querySelector('#audio-wave-wrap');

  let audioScrub = false;
  const waveSeek = e => {
    const rect = audioState.wrap.getBoundingClientRect();
    seek((e.clientX - rect.left) / state.px);
  };
  audioState.wrap.addEventListener('pointerdown', e => {
    audioScrub = true;
    try { audioState.wrap.setPointerCapture(e.pointerId); } catch (_) {}
    waveSeek(e);
  });
  audioState.wrap.addEventListener('pointermove', e => { if (audioScrub) waveSeek(e); });
  audioState.wrap.addEventListener('pointerup', () => { audioScrub = false; });

  for (const tr of TRACKS) {
    const row = document.createElement('div');
    row.className = 'tl-row';
    row.style.height = LANE_H + 'px';
    row.innerHTML = `
      <div class="tl-name">
        <button class="kfbtn" data-kf="${tr.id}" title="在当前时间添加/移除关键帧">◇</button>
        <span class="tlabel">${tr.label}</span>
        <span class="tval" data-id="${tr.id}" title="按住拖动修改数值 · Shift 精细微调 · 自动在当前时间打关键帧">—</span>
        <span class="group-tag">${tr.g}</span>
      </div>
      <div class="tl-lane" data-lane="${tr.id}"></div>`;
    tlContent.appendChild(row);
    laneEls[tr.id] = row.querySelector('.tl-lane');
    makeScrub(row.querySelector('.tval'), tr);
    row.querySelector('.kfbtn').addEventListener('click', () => {
      snapshot();
      const idx = keyIndexAt(tr.id, state.time);
      if (idx >= 0) removeKey(tr.id, idx);
      else upsertKey(tr.id, state.time, currentValue(tr.id));
      renderTimeline(); applyAll(state.time);
    });
  }
  layoutTimeline();
  drawRuler();
  renderDiamonds();
  bindTimelineEvents();
}

function layoutTimeline() {
  const w = trackWidth();
  tlContent.style.width = (NAMES_W + w + 40) + 'px';
  tlContent.style.minHeight = '100%';
  rulerCanvas.width = Math.max(1, Math.round(w));
  rulerCanvas.style.width = w + 'px';
  for (const tr of TRACKS) laneEls[tr.id].style.width = w + 'px';
  if (audioState.wave) {
    audioState.wave.width = Math.max(1, Math.round(w));
    audioState.wave.style.width = w + 'px';
    audioState.wave.height = AUDIO_ROW_H;
  }
  const h = RULER_H + AUDIO_ROW_H + TRACKS.length * LANE_H;
  playheadEl.style.height = h + 'px';
}

function drawRuler() {
  const ctx = rulerCanvas.getContext('2d');
  const w = rulerCanvas.width, h = RULER_H;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#262932'; ctx.fillRect(0, 0, w, h);
  const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 30];
  let step = steps.find(s => s * state.px >= 70) || 30;
  ctx.font = '10px sans-serif';
  ctx.textBaseline = 'top';
  for (let t = 0; t <= state.duration + 1e-6; t += step / 5) {
    const x = t * state.px;
    const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
    ctx.strokeStyle = major ? '#565c6a' : '#3a3f4c';
    ctx.beginPath(); ctx.moveTo(x + 0.5, major ? h - 14 : h - 7); ctx.lineTo(x + 0.5, h); ctx.stroke();
    if (major) {
      ctx.fillStyle = '#9aa0ad';
      ctx.fillText((Math.round(t * 100) / 100) + 's', x + 4, 4);
    }
  }
}

function renderDiamonds() {
  for (const tr of TRACKS) {
    const lane = laneEls[tr.id];
    lane.querySelectorAll('.diamond').forEach(d => d.remove());
    keysOf(tr.id).forEach((k, i) => {
      const d = document.createElement('div');
      d.className = 'diamond' + (k.interp !== 'smooth' ? ' ' + k.interp : '');
      d.style.left = (k.t * state.px) + 'px';
      d.dataset.track = tr.id; d.dataset.index = i;
      d.title = `${tr.label} · ${k.t.toFixed(2)}s = ${(+k.v).toFixed(2)}（${k.interp}）`;
      if (state.selected && state.selected.trackId === tr.id && state.selected.index === i)
        d.classList.add('selected');
      lane.appendChild(d);
    });
  }
}
function renderTimeline() { layoutTimeline(); drawRuler(); renderAudioWave(); renderDiamonds(); updateKfButtons(); syncAudioTime(); }

function renderAudioWave() {
  if (!audioState.ready || !audioState.waveCtx) return;
  const ctx = audioState.waveCtx;
  const w = audioState.wave.width, h = AUDIO_ROW_H;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#15181e'; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#2a3038';
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  const n = audioState.peaks.length;
  const mid = h / 2;
  const amp = h / 2 - 8;
  ctx.fillStyle = '#5aa2f0';
  for (let x = 0; x < w; x++) {
    const idx = Math.min(n - 1, Math.floor(x / w * n));
    const v = audioState.peaks[idx];
    const barH = Math.max(1, v * amp);
    ctx.fillRect(x, mid - barH, 1, barH * 2);
  }
}

function updatePlayhead() {
  playheadEl.style.left = (NAMES_W + state.time * state.px) + 'px';
}
function updateTimeDisplay() {
  const el = document.getElementById('time-display');
  el.textContent = `${state.time.toFixed(2)}s · 帧 ${Math.round(state.time * PREVIEW_FPS)}`;
}

function seek(t, pause = true) {
  state.time = Math.min(state.duration, Math.max(0, t));
  if (pause) setPlaying(false);
  syncAudioTime();
  applyAll(state.time);
}

function bindTimelineEvents() {
  let scrubbing = false;
  rulerCanvas.addEventListener('pointerdown', e => {
    scrubbing = true;
    try { rulerCanvas.setPointerCapture(e.pointerId); } catch (_) {}
    seek(e.offsetX / state.px);
  });
  rulerCanvas.addEventListener('pointermove', e => { if (scrubbing) seek(e.offsetX / state.px); });
  rulerCanvas.addEventListener('pointerup', () => { scrubbing = false; });

  let drag = null;
  tlContent.addEventListener('pointerdown', e => {
    const d = e.target.closest('.diamond');
    if (!d) return;
    e.stopPropagation();
    const trackId = d.dataset.track, index = +d.dataset.index;
    state.selected = { trackId, index };
    tlContent.querySelectorAll('.diamond.selected').forEach(x => x.classList.remove('selected'));
    d.classList.add('selected');
    beginGesture();
    drag = { trackId, index, moved: false };
    try { d.setPointerCapture(e.pointerId); } catch (_) {}
  });
  tlContent.addEventListener('pointermove', e => {
    if (!drag) return;
    const lane = laneEls[drag.trackId];
    const rect = lane.getBoundingClientRect();
    let t = (e.clientX - rect.left) / state.px;
    t = Math.min(state.duration, Math.max(0, t));
    const k = keysOf(drag.trackId)[drag.index];
    if (!k) { drag = null; return; }
    if (!drag.moved) snapshot();
    k.t = t;
    drag.moved = true;
    const el = lane.querySelector(`.diamond[data-index="${drag.index}"]`);
    if (el) el.style.left = (t * state.px) + 'px';
  });
  tlContent.addEventListener('pointerup', () => {
    if (drag && drag.moved) {
      keysOf(drag.trackId).sort((a, b) => a.t - b.t);
      renderTimeline(); applyAll(state.time);
    }
    drag = null;
  });
  tlContent.addEventListener('dblclick', e => {
    const d = e.target.closest('.diamond');
    if (d) { openKfEditor(d.dataset.track, +d.dataset.index, e.clientX, e.clientY); return; }
    const lane = e.target.closest('.tl-lane');
    if (lane) {
      const rect = lane.getBoundingClientRect();
      const t = Math.min(state.duration, Math.max(0, (e.clientX - rect.left) / state.px));
      const id = lane.dataset.lane;
      snapshot();
      upsertKey(id, t, evalTrack(id, t));
      seek(t);
      renderTimeline();
    }
  });

  let laneScrub = null;
  const laneSeek = (e, lane) => {
    const rect = lane.getBoundingClientRect();
    seek((e.clientX - rect.left) / state.px);
  };
  tlContent.addEventListener('pointerdown', e => {
    if (e.target.closest('.diamond')) return;
    if (e.target.closest('#audio-wave-wrap')) return;
    const lane = e.target.closest('.tl-lane');
    if (!lane) return;
    laneScrub = lane;
    try { lane.setPointerCapture(e.pointerId); } catch (_) {}
    laneSeek(e, lane);
  });
  tlContent.addEventListener('pointermove', e => {
    if (laneScrub) laneSeek(e, laneScrub);
  });
  tlContent.addEventListener('pointerup', () => { laneScrub = null; });
}

// --- 关键帧编辑弹窗 ---
const kfEditor = document.getElementById('kf-editor');
let editing = null;
function openKfEditor(trackId, index, x, y) {
  const k = keysOf(trackId)[index];
  if (!k) return;
  editing = { trackId, index };
  document.getElementById('kf-track').value = TRACK_MAP[trackId].label;
  document.getElementById('kf-time').value = k.t.toFixed(2);
  document.getElementById('kf-value').value = (+k.v).toFixed(3);
  document.getElementById('kf-interp').value = k.interp;
  kfEditor.style.display = 'block';
  const px = Math.min(x, window.innerWidth - 230);
  const py = Math.min(y, window.innerHeight - 220);
  kfEditor.style.left = px + 'px'; kfEditor.style.top = py + 'px';
  state.selected = { trackId, index };
  renderDiamonds();
}
function closeKfEditor() { kfEditor.style.display = 'none'; editing = null; }
document.getElementById('kf-time').addEventListener('change', e => {
  if (!editing) return;
  snapshot();
  const k = keysOf(editing.trackId)[editing.index];
  k.t = Math.min(state.duration, Math.max(0, parseFloat(e.target.value) || 0));
  keysOf(editing.trackId).sort((a, b) => a.t - b.t);
  editing.index = keysOf(editing.trackId).indexOf(k);
  state.selected = { ...editing };
  renderTimeline(); applyAll(state.time);
});
document.getElementById('kf-value').addEventListener('change', e => {
  if (!editing) return;
  snapshot();
  const tr = TRACK_MAP[editing.trackId];
  let val = parseFloat(e.target.value) || 0;
  val = Math.min(tr.max, Math.max(tr.min, val));
  keysOf(editing.trackId)[editing.index].v = tr.integer ? Math.round(val) : val;
  renderTimeline(); applyAll(state.time);
});
document.getElementById('kf-interp').addEventListener('change', e => {
  if (!editing) return;
  snapshot();
  keysOf(editing.trackId)[editing.index].interp = e.target.value;
  renderTimeline(); applyAll(state.time);
});
document.getElementById('kf-delete').addEventListener('click', () => {
  if (!editing) return;
  snapshot();
  removeKey(editing.trackId, editing.index);
  closeKfEditor(); renderTimeline(); applyAll(state.time);
});
document.getElementById('kf-close').addEventListener('click', closeKfEditor);

// --- 工具栏 ---
function setPlaying(on) {
  state.playing = on;
  document.getElementById('btn-play').textContent = on ? '⏸' : '▶';
  if (audioState.ready && audioState.el) {
    if (on) {
      if (audioState.el.ended) audioState.el.currentTime = state.time;
      audioState.el.play().catch(() => {});
    } else {
      audioState.el.pause();
    }
  }
}
document.getElementById('btn-play').addEventListener('click', () => setPlaying(!state.playing));
document.getElementById('btn-start').addEventListener('click', () => seek(0));
document.getElementById('btn-end').addEventListener('click', () => seek(state.duration));
document.getElementById('btn-prevf').addEventListener('click', () => seek(state.time - 1 / PREVIEW_FPS));
document.getElementById('btn-nextf').addEventListener('click', () => seek(state.time + 1 / PREVIEW_FPS));
document.getElementById('chk-loop').addEventListener('change', e => { state.loop = e.target.checked; });
document.getElementById('inp-duration').addEventListener('change', e => {
  snapshot();
  state.duration = Math.min(300, Math.max(1, parseFloat(e.target.value) || 14));
  for (const tr of TRACKS) keysOf(tr.id).forEach(k => { k.t = Math.min(k.t, state.duration); });
  renderTimeline(); seek(Math.min(state.time, state.duration));
});
document.getElementById('inp-zoom').addEventListener('input', e => {
  state.px = +e.target.value; renderTimeline(); updatePlayhead();
});
document.getElementById('btn-key-all').addEventListener('click', () => {
  snapshot();
  for (const tr of TRACKS) upsertKey(tr.id, state.time, currentValue(tr.id));
  renderTimeline(); applyAll(state.time);
});
document.getElementById('btn-del-key').addEventListener('click', () => {
  if (state.selected) {
    snapshot();
    removeKey(state.selected.trackId, state.selected.index);
    renderTimeline(); applyAll(state.time);
  }
});
document.getElementById('btn-clear-all').addEventListener('click', () => {
  const total = TRACKS.reduce((s, tr) => s + keysOf(tr.id).length, 0);
  if (total === 0) { flashHint('当前没有任何关键帧'); return; }
  if (!confirm(`确认删除全部 ${total} 个关键帧？（可用 ⌘Z 撤回）`)) return;
  snapshot();
  for (const tr of TRACKS) keysOf(tr.id).length = 0;
  state.selected = null;
  closeKfEditor();
  renderTimeline();
  applyAll(state.time);
  flashHint(`已清空全部 ${total} 个关键帧`);
});

window.addEventListener('keydown', e => {
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.code === 'KeyZ') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); return; }
  if (mod && e.code === 'KeyY') { e.preventDefault(); redo(); return; }
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!state.playing); }
  else if (e.key === 'ArrowLeft') seek(state.time - 1 / PREVIEW_FPS);
  else if (e.key === 'ArrowRight') seek(state.time + 1 / PREVIEW_FPS);
  else if (e.key === 'Delete' || e.key === 'Backspace') {
    if (state.selected) { snapshot(); removeKey(state.selected.trackId, state.selected.index); renderTimeline(); applyAll(state.time); }
  } else if (e.key === 'Escape') closeKfEditor();
});

// ---------------------------------------------------------------------------
// 9. 视角切换
// ---------------------------------------------------------------------------
const btnView = document.getElementById('btn-view');
const viewBadge = document.getElementById('view-badge');
function setView(mode) {
  state.view = mode;
  const free = mode === 'free';
  controls.enabled = free;
  btnView.textContent = free ? '🖐 自由视角' : '🎥 摄像机视角';
  btnView.classList.toggle('on', !free);
  btnSyncCam.classList.toggle('on', free);
  viewBadge.textContent = free
    ? '自由视角 · 拖拽旋转 / 滚轮缩放 / 右键平移（不影响关键帧）'
    : '摄像机视角 · 按关键帧动画渲染';
  applyAll(state.time);
}
btnView.addEventListener('click', () => setView(state.view === 'camera' ? 'free' : 'camera'));

// ---------------------------------------------------------------------------
// 9b. 同步自由视角 → 当前时间关键帧
//     在自由视角摆好机位（旋转/缩放/平移视觉中心）后，点击按钮把当前视口
//     的相机位置 + 视觉中心写入播放头时间的关键帧；若取消"看向视觉中心"
//     则额外写入旋转欧拉角。
// ---------------------------------------------------------------------------
const btnSyncCam = document.getElementById('btn-sync-cam');
let syncToastTimer = null;
function clampTrack(id, v) {
  const tr = TRACK_MAP[id];
  return Math.min(tr.max, Math.max(tr.min, v));
}
function syncFreeViewToKeys() {
  snapshot();
  const t = state.time;
  const set = (id, v) => upsertKey(id, t, clampTrack(id, v));
  set('camX', camera.position.x);
  set('camY', camera.position.y);
  set('camZ', camera.position.z);
  set('tgtX', controls.target.x);
  set('tgtY', controls.target.y);
  set('tgtZ', controls.target.z);
  if (!state.lookAtTarget) {
    const eul = new THREE.Euler(0, 0, 0, 'YXZ').setFromQuaternion(camera.quaternion);
    set('rotX', THREE.MathUtils.radToDeg(eul.x));
    set('rotY', THREE.MathUtils.radToDeg(eul.y));
    set('rotZ', THREE.MathUtils.radToDeg(eul.z));
  }
  renderTimeline();
  applyAll(state.time);
  viewBadge.textContent = `📌 已同步自由视角机位 → 关键帧 @ ${t.toFixed(2)}s${state.lookAtTarget ? '' : '（含旋转）'}`;
  clearTimeout(syncToastTimer);
  syncToastTimer = setTimeout(() => {
    viewBadge.textContent = state.view === 'free'
      ? '自由视角 · 拖拽旋转 / 滚轮缩放 / 右键平移（不影响关键帧）'
      : '摄像机视角 · 按关键帧动画渲染';
  }, 2200);
}
btnSyncCam.addEventListener('click', syncFreeViewToKeys);

// --- 撤销 / 重做按钮 ---
document.getElementById('btn-undo').addEventListener('click', undo);
document.getElementById('btn-redo').addEventListener('click', redo);
updateUndoButtons();

// ---------------------------------------------------------------------------
// 10. 导出
// ---------------------------------------------------------------------------
const exportModal = document.getElementById('export-modal');
const expRes = document.getElementById('exp-res');
const expCustomRow = document.getElementById('exp-custom-row');
expRes.addEventListener('change', () => {
  expCustomRow.style.display = expRes.value === 'custom' ? 'flex' : 'none';
});
document.getElementById('exp-format').addEventListener('change', () => {
  const f = document.getElementById('exp-format').value;
  document.getElementById('exp-mix-row').style.display = (f === 'webm' && audioState.ready) ? 'flex' : 'none';
});
document.getElementById('btn-export').addEventListener('click', () => {
  const mixOk = audioState.ready;
  document.getElementById('exp-mix-row').style.display = mixOk ? 'flex' : 'none';
  document.getElementById('exp-mix').checked = mixOk;
  document.getElementById('exp-range').value = mixOk
    ? `口播 ${audioState.duration.toFixed(1)}s · 动画 ${state.duration}s — 混音导出为实时录制，时长以口播为准`
    : `0 – ${state.duration}s（共 ${Math.round(state.duration * PREVIEW_FPS)} 帧预览，导出帧数 = 时长 × 帧率）`;
  document.getElementById('export-status').textContent = '';
  document.getElementById('export-progress').style.display = 'none';
  exportModal.style.display = 'flex';
});
document.getElementById('exp-cancel').addEventListener('click', () => {
  if (exporting) { exportCancelled = true; }
  else exportModal.style.display = 'none';
});

let exporting = false, exportCancelled = false;

function downloadBlob(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

async function exportLiveVoice(w, h, fps) {
  const bar = document.querySelector('#export-progress i');
  const status = document.getElementById('export-status');
  const prog = document.getElementById('export-progress');
  prog.style.display = 'block';
  exporting = true; exportCancelled = false;
  setPlaying(false);

  const canvas = renderer.domElement;
  const oldW = canvas.width, oldH = canvas.height;
  const oldAspect = camera.aspect;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();

  try {
    const stream = canvas.captureStream(fps);
    const audioTracks = audioState.el.captureStream().getAudioTracks();
    if (!audioTracks.length) throw new Error('无法捕获口播音频轨');
    audioTracks.forEach(t => stream.addTrack(t));
    const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus')
      ? 'video/webm;codecs=vp8,opus'
      : (MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : null);
    if (!mime) throw new Error('当前浏览器不支持 MediaRecorder WebM');

    const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12e6, audioBitsPerSecond: 160e3 });
    const chunks = [];
    rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
    const stopped = new Promise(res => { rec.onstop = res; });

    rec.start(100);
    state.exportLive = true;
    audioState.el.currentTime = 0;
    seek(0, false);
    setPlaying(true);

    while (state.exportLive) {
      await new Promise(r => setTimeout(r, 100));
      if (exportCancelled || audioState.el.ended || state.time >= state.duration - 1e-4) state.exportLive = false;
      status.textContent = `实时录制中 · 口播 ${audioState.el.currentTime.toFixed(2)}s / ${audioState.duration.toFixed(1)}s（音画同步）`;
      bar.style.width = Math.min(100, (audioState.el.currentTime / state.duration * 100).toFixed(1)) + '%';
    }
    setPlaying(false);
    audioState.el.pause();
    rec.stop();
    await stopped;

    if (exportCancelled) {
      status.textContent = '已取消导出。';
    } else {
      const blob = new Blob(chunks, { type: mime });
      downloadBlob(blob, `axis_switching_voice_${w}x${h}_${fps}fps.webm`);
      status.textContent = `✅ 已导出带口播的 WebM 视频（${w}×${h}，${blob.size / 1048576 > 1 ? (blob.size / 1048576).toFixed(1) + ' MB' : Math.round(blob.size / 1024) + ' KB'}）`;
    }
  } catch (err) {
    status.textContent = '混音导出失败：' + err.message;
  } finally {
    state.exportLive = false;
    renderer.setSize(oldW, oldH, false);
    camera.aspect = oldAspect;
    camera.updateProjectionMatrix();
    exporting = false;
    applyAll(state.time);
  }
}

document.getElementById('exp-start').addEventListener('click', async () => {
  if (exporting) return;
  let w, h;
  if (expRes.value === 'custom') {
    w = Math.max(16, +document.getElementById('exp-w').value || 1920);
    h = Math.max(16, +document.getElementById('exp-h').value || 1080);
  } else { [w, h] = expRes.value.split('x').map(Number); }
  const fps = +document.getElementById('exp-fps').value;
  const format = document.getElementById('exp-format').value;
  const mix = document.getElementById('exp-mix').checked;

  if (format === 'webm' && mix && audioState.ready) {
    await exportLiveVoice(w, h, fps);
    return;
  }
  const frames = Math.max(1, Math.round(state.duration * fps));

  const bar = document.querySelector('#export-progress i');
  const status = document.getElementById('export-status');
  const prog = document.getElementById('export-progress');
  prog.style.display = 'block';
  exporting = true; exportCancelled = false;
  setPlaying(false);

  const canvas = renderer.domElement;
  const oldW = canvas.width, oldH = canvas.height;
  const oldAspect = camera.aspect;

  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();

  let zip = null, video = null;
  if (format === 'png') zip = new JSZip();
  else video = new Whammy.Video(fps);

  const pad = n => String(n).padStart(4, '0');
  try {
    for (let i = 0; i < frames; i++) {
      if (exportCancelled) break;
      const t = i / fps;
      applyAll(t, true);
      controls.update();
      renderer.render(scene, camera);
      if (format === 'png') {
        const blob = await new Promise(r => canvas.toBlob(r, 'image/png'));
        zip.file(`frame_${pad(i)}.png`, blob);
      } else {
        video.add(canvas.toDataURL('image/webp', 0.92));
      }
      bar.style.width = ((i + 1) / frames * 100).toFixed(1) + '%';
      status.textContent = `渲染帧 ${i + 1} / ${frames}（t = ${t.toFixed(2)}s）`;
      await new Promise(r => setTimeout(r, 0));
    }
    if (!exportCancelled) {
      status.textContent = '正在封装文件…';
      await new Promise(r => setTimeout(r, 30));
      if (format === 'png') {
        const blob = await zip.generateAsync({ type: 'blob' }, m => {
          bar.style.width = (m.percent).toFixed(1) + '%';
        });
        downloadBlob(blob, `axis_switching_${w}x${h}_${fps}fps_序列帧.zip`);
        status.textContent = `✅ 已导出 ${frames} 帧 PNG 序列（${w}×${h} @ ${fps}fps）`;
      } else {
        const blob = await new Promise(res => video.compile(false, res));
        downloadBlob(blob, `axis_switching_${w}x${h}_${fps}fps.webm`);
        status.textContent = `✅ 已导出 WebM 视频（${w}×${h} @ ${fps}fps，${frames} 帧）`;
      }
    } else {
      status.textContent = '已取消导出。';
    }
  } catch (err) {
    status.textContent = '导出失败：' + err.message;
  } finally {
    renderer.setSize(oldW, oldH, false);
    camera.aspect = oldAspect;
    camera.updateProjectionMatrix();
    exporting = false;
    applyAll(state.time);
  }
});

// ---------------------------------------------------------------------------
// 11. 布局 / 主循环
// ---------------------------------------------------------------------------
function resize() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

{
  const splitter = document.getElementById('splitter');
  const timeline = document.getElementById('timeline');
  let startY = 0, startH = 0, dragOn = false;
  splitter.addEventListener('pointerdown', e => {
    dragOn = true; startY = e.clientY; startH = timeline.offsetHeight;
    try { splitter.setPointerCapture(e.pointerId); } catch (_) {}
  });
  splitter.addEventListener('pointermove', e => {
    if (!dragOn) return;
    const nh = Math.min(window.innerHeight - 140, Math.max(120, startH + (startY - e.clientY)));
    timeline.style.height = nh + 'px';
    timeline.style.flexBasis = nh + 'px';
    resize();
  });
  splitter.addEventListener('pointerup', () => { dragOn = false; });
}

const clock = new THREE.Clock();
function loop() {
  requestAnimationFrame(loop);
  const dt = Math.min(clock.getDelta(), 0.1);
  if (state.playing) {
    if (state.exportLive && audioState.ready) {
      state.time = Math.min(state.duration, audioState.el.currentTime);
      if (audioState.el.ended) { state.time = state.duration; setPlaying(false); }
    } else {
      state.time += dt;
      if (state.time >= state.duration) {
        if (state.loop) {
          state.time = 0;
          if (audioState.ready && audioState.el) audioState.el.currentTime = 0;
        }
        else { state.time = state.duration; setPlaying(false); }
      }
      if (audioState.ready && audioState.el && audioState.el.paused && !audioState.el.ended)
        audioState.el.play().catch(() => {});
    }
  }
  syncAudioTime();
  controls.update();
  applyAll(state.time);
  renderer.render(scene, camera);
}

buildTimeline();
resize();
applyAll(0);
loop();
