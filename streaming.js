/**
 * Enhanced Gaussian Splatting Streaming with I-frame/P-frame Compression
 * Integrates GoP-based temporal compression for dynamic scenes
 */

// Import essential functions and globals from main.js
var global_log_1 = {"parse":[], "vertex":[]};

// Camera and view matrix utilities
let cameras = [
    {
        "position": [
            0.007232260564530732,
            0.809987791156739,
            5.33903731414556
        ],
        "rotation": [
            [
                0.9999727319388764,
                -0.0007335337134094191,
                0.007348285990037004
            ],
            [
                -0.0008200432778977561,
                -0.9999303163911514,
                0.011776667224411872
            ],
            [
                0.007339135352509657,
                -0.011782372010060363,
                -0.9999036517595554
            ]
        ],
        "fy": 1834.5526065144063,
        "fx": 1834.8835144895522,
    }
];

let camera = cameras[0];

// Matrix utility functions
function getProjectionMatrix(fx, fy, width, height) {
    const znear = 0.2;
    const zfar = 200;
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, -(2 * fy) / height, 0, 0],
        [0, 0, zfar / (zfar - znear), 1],
        [0, 0, -(zfar * znear) / (zfar - znear), 0],
    ].flat();
}

function getViewMatrix(camera) {
    const R = camera.rotation.flat();
    const t = [
        camera.position[0],
        camera.position[1],
        camera.position[2]];
    
    const camToWorld = [
        [R[0], R[1], R[2], 0],
        [R[3], R[4], R[5], 0],
        [R[6], R[7], R[8], 0],
        [t[0], t[1], t[2], 1],
    ].flat();
    return camToWorld;
}

function getViewMatrixDefault(camera) {
    return defaultViewMatrix;
}

function multiply4(a, b) {
    return [
        b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
        b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
        b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
        b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
        b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
        b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
        b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
        b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
        b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
        b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
        b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
}

function invert4(a) {
    let b00 = a[0] * a[5] - a[1] * a[4];
    let b01 = a[0] * a[6] - a[2] * a[4];
    let b02 = a[0] * a[7] - a[3] * a[4];
    let b03 = a[1] * a[6] - a[2] * a[5];
    let b04 = a[1] * a[7] - a[3] * a[5];
    let b05 = a[2] * a[7] - a[3] * a[6];
    let b06 = a[8] * a[13] - a[9] * a[12];
    let b07 = a[8] * a[14] - a[10] * a[12];
    let b08 = a[8] * a[15] - a[11] * a[12];
    let b09 = a[9] * a[14] - a[10] * a[13];
    let b10 = a[9] * a[15] - a[11] * a[13];
    let b11 = a[10] * a[15] - a[11] * a[14];
    let det =
        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ];
}

function rotate4(a, rad, x, y, z) {
    let len = Math.hypot(x, y, z);
    x /= len;
    y /= len;
    z /= len;
    let s = Math.sin(rad);
    let c = Math.cos(rad);
    let t = 1 - c;
    let b00 = x * x * t + c;
    let b01 = y * x * t + z * s;
    let b02 = z * x * t - y * s;
    let b10 = x * y * t - z * s;
    let b11 = y * y * t + c;
    let b12 = z * y * t + x * s;
    let b20 = x * z * t + y * s;
    let b21 = y * z * t - x * s;
    let b22 = z * z * t + c;
    return [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
        ...a.slice(12, 16),
    ];
}

function translate4(a, x, y, z) {
    return [
        ...a.slice(0, 12),
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ];
}

// Global variables
var SLICE_NUM
var TOTAL_CAP
var SLICE_CAP
var SH_DEGREE
var STREAM_ROW_LENGTH
var VERTEX_ROW_LENGTH
var defaultViewMatrix
var MAX_FRAME
var MINIMAL_BW = 22
var splatData
var loadedFrame

function setup_consts(config) {
    MAX_FRAME = config.MAX_FRAME;
    SLICE_NUM = config.SLICE_NUM;
    TOTAL_CAP = config.TOTAL_CAP;
    SLICE_CAP = Math.ceil(TOTAL_CAP / SLICE_NUM);
    STREAM_ROW_LENGTH = config.STREAM_ROW_LENGTH;
    SH_DEGREE = config.SH_DEGREE;
    console.log("STREAM_ROW_LENGTH", STREAM_ROW_LENGTH);
    VERTEX_ROW_LENGTH = config.VERTEX_ROW_LENGTH;
    defaultViewMatrix = config.INIT_VIEW;
    let R = [
        [defaultViewMatrix[0], defaultViewMatrix[1], defaultViewMatrix[2]],
        [defaultViewMatrix[4], defaultViewMatrix[5], defaultViewMatrix[6]],
        [defaultViewMatrix[8], defaultViewMatrix[9], defaultViewMatrix[10]],
    ]
    let t = [
        defaultViewMatrix[12],
        defaultViewMatrix[13],
        defaultViewMatrix[14],
    ]
    cameras[0].rotation = R;
    cameras[0].position = t;
    cameras[0].fx = config.fx;
    cameras[0].fy = config.fy;
    MINIMAL_BW = Math.ceil(config.STREAM_ROW_LENGTH * config.TOTAL_CAP * config.FPS / 1e6 / SLICE_NUM);
    console.log("MINIMAL_BW", MINIMAL_BW);
    const min_bw_ele = document.getElementById("min_bw");
    if (min_bw_ele) min_bw_ele.innerHTML = MINIMAL_BW;
}

function GS_TO_VERTEX_COMPACT(gs, full_gs=false) {
    // input list of gs objects
    // output buffer of binary data
    let start = Date.now();
    const buffer = new ArrayBuffer(gs.length * VERTEX_ROW_LENGTH);
    const vertexCount = gs.length;
    console.time("build buffer");
    if (full_gs) {
        // for the full gs, sort by end frame
        gs.sort((a, b) => a.end_frame - b.end_frame);
    } else {
        // for the slice gs, sort by start frame
        gs.sort((a, b) => a.start_frame - b.start_frame);
    }
    let curFrame = gs.length > 0 ? gs[0].start_frame : 0;
    let curSliceStart = 0;
    let frame_spans = [];
    for (let j = 0; j < vertexCount; j++) {
        let attrs = gs[j];
        if (! full_gs) { // for slice
            if (attrs.start_frame != curFrame || j == vertexCount - 1) {
                frame_spans.push(
                    {
                        frame: curFrame,
                        from: curSliceStart,
                        to: j,
                        total: j - curSliceStart -1
                    }
                )
                curFrame = gs[j].start_frame;
                curSliceStart = j;
            }
        } else {
            frame_spans.push(attrs.end_frame)
        }
        // memory pointer we need to fill
        const position = new Float32Array(buffer, j * VERTEX_ROW_LENGTH, 3);
        const scales = new Float32Array(buffer, j * VERTEX_ROW_LENGTH + 4 * 3, 3);
        const rgba = new Uint8ClampedArray(
            buffer,
            j * VERTEX_ROW_LENGTH + 4 * 3 + 4 * 3,
            4,
        );
        const rot = new Uint8ClampedArray(
            buffer,
            j * VERTEX_ROW_LENGTH + 4 * 3 + 4 * 3 + 4,
            4,
        );

        rot[0] = attrs.rotation[0];
        rot[1] = attrs.rotation[1];
        rot[2] = attrs.rotation[2];
        rot[3] = attrs.rotation[3];
        scales[0] = attrs.scaling[0];
        scales[1] = attrs.scaling[1];
        scales[2] = attrs.scaling[2];
        position[0] = attrs.xyz[0];
        position[1] = attrs.xyz[1];
        position[2] = attrs.xyz[2];
        rgba[0] = attrs.color[0];
        rgba[1] = attrs.color[1];
        rgba[2] = attrs.color[2];
        rgba[3] = attrs.opacity;
        
    }
    console.timeEnd("build buffer");
    let end = Date.now();
    if (!full_gs) global_log_1["vertex"].push(end - start);
    return {all:new Uint8Array(buffer), spans:frame_spans};
}

function PARSE_RAW_BYTES_COMPACT(arrayLike) {
    let start = Date.now();
    const view = new DataView(arrayLike.buffer, arrayLike.byteOffset, arrayLike.byteLength);
    const jsonObjects = [];
    const sizeOfObject = STREAM_ROW_LENGTH; // total bytes for one object
    
    for (let offset = 0; offset < arrayLike.byteLength; offset += sizeOfObject) {
        const start_frame = view.getUint16(offset, false); // true for little-endian
        const end_frame = view.getUint16(offset + 2, false);

        const xyz = [
            view.getFloat32(offset + 4, false),
            view.getFloat32(offset + 8, false),
            view.getFloat32(offset + 12, false),
        ];
        
        const color = [
            view.getUint8(offset + 16, false),
            view.getUint8(offset + 17, false),
            view.getUint8(offset + 18, false),
        ];
        const opacity = view.getUint8(offset + 19, false);

        // we dont really need f_rest
        
        const scaling = [
            view.getFloat32(offset + 20, false),
            view.getFloat32(offset + 24, false),
            view.getFloat32(offset + 28, false),
        ];
        
        const rotation = [
            view.getUint8(offset + 32, false),
            view.getUint8(offset + 33, false),
            view.getUint8(offset + 34, false),
            view.getUint8(offset + 35, false),
        ];
        
        
        jsonObjects.push({
            start_frame,
            end_frame,
            xyz,
            color,
            opacity,
            scaling,
            rotation,
        });
    }
    let end = Date.now();
    global_log_1["parse"].push(end - start);
    return jsonObjects;
}

function average(arr){
    return arr.reduce((a,b) => a+b, 0) / arr.length;
}

// WebGL Shaders
const vertexShaderSource = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;

in vec2 position;
in int index;

out vec4 vColor;
out vec2 vPosition;

void main () {
    uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
    vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1);
    vec4 pos2d = projection * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
    vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z), 
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z), 
        0., 0., 0.
    );

    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;
    vPosition = position;

    vec2 vCenter = vec2(pos2d) / pos2d.w;
    gl_Position = vec4(
        vCenter 
        + position.x * majorAxis / viewport 
        + position.y * minorAxis / viewport, 0.0, 1.0);

}
`.trim();

const fragmentShaderSource = `
#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vPosition;

out vec4 fragColor;

void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    fragColor = vec4(B * vColor.rgb, B);
}

`.trim();

// Progress update functions
function update_curframe(cur_frame) {
    let curFrameElem = document.getElementById("progress-current");
    if (curFrameElem && MAX_FRAME > 0) {
        let curFramePercent = (cur_frame / (MAX_FRAME-1)) * 100;
        curFrameElem.style.width = curFramePercent + "%";
    }
}

function update_buffered(loaded_frame) {
    let bufferedElem = document.getElementById("progress-buffered");
    if (bufferedElem && MAX_FRAME > 0) {
        let bufferedPercent = ((loaded_frame+1) / (MAX_FRAME-1)) * 100;
        bufferedElem.style.width = bufferedPercent + "%";
    }
}

var lastFrame = NaN;
var videoAvgFps = 30;
function update_FPS() {
    if (isNaN(lastFrame)){
        lastFrame = Date.now();
        return;
    }
    let now = Date.now();
    let currentFps = 1000 / (now - lastFrame) || 0;
    videoAvgFps = videoAvgFps * 0.9 + currentFps * 0.1;

    const curFPS= document.getElementById("FPS");
    if (curFPS) curFPS.innerText = Math.ceil(videoAvgFps);
    lastFrame = now;
}

function setup_dropdown(target) {
    const dropdown = document.getElementById("scene_name");
    if (dropdown) {
        for (let i = 0; i < dropdown.options.length; i++) {
            if (dropdown.options[i].value == target) {
                dropdown.selectedIndex = i;
                break;
            }
        }
        dropdown.addEventListener("change", (e) => {
            location.search = `?target=${e.target.value}`;
        });
    }
}

// GoP Configuration
var GOP_SIZE = 24; // I-frame every 24 frames (1 second at 24fps)
var COMPRESSION_ENABLED = true;
var I_FRAME_QUALITY = 1.0;
var P_FRAME_COMPRESSION_RATIO = 0.3;

// GoP State Management
var iFrameBuffer = null; // Store I-frame Gaussians for P-frame reconstruction
var currentGoP = 0;
var frameTypeBuffer = new Map(); // Cache frame type information
var compressionStats = {
    totalFrames: 0,
    iFrames: 0,
    pFrames: 0,
    totalBandwidth: 0,
    savedBandwidth: 0
};

/**
 * Frame type detection
 */
function isIFrame(frameNumber) {
    return frameNumber === 0 || frameNumber % GOP_SIZE === 0;
}

function getGoPNumber(frameNumber) {
    return Math.floor(frameNumber / GOP_SIZE);
}

/**
 * Enhanced frame event structure for I/P frames
 */
function createFrameEvent(frameNum, data, type, isIFrame = false, reusableIndices = null, residualData = null) {
    return {
        frame: frameNum,
        sliceId: (frameNum - 1) % SLICE_NUM,
        data: data,
        type: type,
        isIFrame: isIFrame,
        reusableIndices: reusableIndices,
        residualData: residualData,
        gopStart: frameNum % GOP_SIZE === 0,
        dataSize: data ? data.length : 0
    };
}

/**
 * Parse P-frame compressed binary data
 * Format: [4 bytes: reusable_count][reusable_count * 4 bytes: indices][remaining: residual_data]
 */
function parsePFrameData(compressedBytes) {
    if (compressedBytes.length < 4) {
        console.error("Invalid P-frame data: insufficient length");
        return { reusableIndices: [], residualGaussians: [], isValid: false };
    }
    
    try {
        const view = new DataView(compressedBytes.buffer, compressedBytes.byteOffset, compressedBytes.byteLength);
        
        // Read reusable indices count (first 4 bytes)
        const reusableCount = view.getUint32(0, true); // little-endian
        console.log(`P-frame parsing: ${reusableCount} reusable indices`);
        
        // Validate reusable count
        if (reusableCount > 1000000) { // Sanity check
            console.error("Invalid reusable count:", reusableCount);
            return { reusableIndices: [], residualGaussians: [], isValid: false };
        }
        
        // Read reusable indices
        let reusableIndices = [];
        for (let i = 0; i < reusableCount; i++) {
            const indexOffset = 4 + i * 4;
            if (indexOffset + 4 > compressedBytes.length) {
                console.error("P-frame data truncated at indices");
                return { reusableIndices: [], residualGaussians: [], isValid: false };
            }
            const index = view.getUint32(indexOffset, true);
            reusableIndices.push(index);
        }
        
        // Read residual Gaussians
        const residualOffset = 4 + reusableCount * 4;
        const residualBytes = compressedBytes.slice(residualOffset);
        const residualGaussians = PARSE_RAW_BYTES_COMPACT(residualBytes);
        
        console.log(`P-frame parsed: ${reusableIndices.length} reusable + ${residualGaussians.length} residual`);
        
        return {
            reusableIndices: reusableIndices,
            residualGaussians: residualGaussians,
            isValid: true,
            compressionRatio: residualGaussians.length / (reusableCount + residualGaussians.length)
        };
        
    } catch (error) {
        console.error("Error parsing P-frame data:", error);
        return { reusableIndices: [], residualGaussians: [], isValid: false };
    }
}

/**
 * Reconstruct P-frame by combining I-frame reusable Gaussians with residual data
 */
function createReconstruction(iFrameGaussians, reusableIndices, residualGaussians) {
    if (!iFrameGaussians || !Array.isArray(iFrameGaussians)) {
        console.error("Invalid I-frame buffer for P-frame reconstruction");
        return residualGaussians || [];
    }
    
    console.log(`Reconstructing P-frame: ${reusableIndices.length} reusable + ${residualGaussians.length} residual`);
    
    // Extract reusable Gaussians from I-frame using indices
    let reusableGaussians = [];
    for (let idx of reusableIndices) {
        if (idx >= 0 && idx < iFrameGaussians.length) {
            // Deep copy to avoid reference issues
            reusableGaussians.push({
                ...iFrameGaussians[idx],
                start_frame: residualGaussians.length > 0 ? residualGaussians[0].start_frame : iFrameGaussians[idx].start_frame,
                end_frame: residualGaussians.length > 0 ? residualGaussians[0].end_frame : iFrameGaussians[idx].end_frame
            });
        } else {
            console.warn(`Invalid reusable index: ${idx} (I-frame has ${iFrameGaussians.length} Gaussians)`);
        }
    }
    
    // Combine reusable + residual Gaussians
    let combinedGaussians = [...reusableGaussians, ...residualGaussians];
    
    console.log(`P-frame reconstruction complete: ${combinedGaussians.length} total Gaussians`);
    
    // Update compression statistics
    compressionStats.totalFrames++;
    compressionStats.pFrames++;
    compressionStats.savedBandwidth += (iFrameGaussians.length - residualGaussians.length) * STREAM_ROW_LENGTH;
    
    return combinedGaussians;
}

/**
 * Store I-frame data for future P-frame reconstructions
 */
function storeIFrameBuffer(gaussians, frameNumber) {
    iFrameBuffer = gaussians.map(g => ({...g})); // Deep copy
    currentGoP = getGoPNumber(frameNumber);
    frameTypeBuffer.set(frameNumber, 'I');
    
    // Update compression statistics
    compressionStats.totalFrames++;
    compressionStats.iFrames++;
    compressionStats.totalBandwidth += gaussians.length * STREAM_ROW_LENGTH;
    
    console.log(`I-frame #${frameNumber} stored: ${gaussians.length} Gaussians, GoP ${currentGoP}`);
}

/**
 * Enhanced streaming main function with I/P frame support
 */
async function enhancedStreamingMain(config) {
    // Setup GoP configuration from config
    GOP_SIZE = config.GOP_SIZE || 24;
    COMPRESSION_ENABLED = config.COMPRESSION_ENABLED !== false;
    
    console.log(`GoP Streaming initialized: GOP_SIZE=${GOP_SIZE}, COMPRESSION=${COMPRESSION_ENABLED}`);
    
    // Initialize original demo components
    viewMatrix = defaultViewMatrix;
    let carousel = false;
    
    try {
        viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
        carousel = false;
    } catch (err) {}
    
    const url = config.MODEL_URL;
    const req = await fetch(url, {
        mode: "cors",
        credentials: "omit",
    });
    
    console.log(req);
    if (req.status != 200)
        throw new Error(req.status + " Unable to load " + req.url);

    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    const reader = req.body.getReader();

    const downsample = 1;
    
    // Enhanced worker creation with GoP support
    const worker = new Worker(
        URL.createObjectURL(
            new Blob([
                "(",
                createEnhancedWorker.toString(),
                `)(self, ${SLICE_CAP}, ${SLICE_NUM}, ${GOP_SIZE})`
            ], {
                type: "application/javascript",
            }),
        ),
    );

    // Initialize WebGL components (same as original)
    const canvas = document.getElementById("canvas");
    const fps = document.getElementById("fps");
    const camid = document.getElementById("camid");

    let projectionMatrix;

    const gl = canvas.getContext("webgl2", {
        antialias: false,
    });

    // WebGL setup (same as original)
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(vertexShader));

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(fragmentShader));

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        console.error(gl.getProgramInfoLog(program));

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    // WebGL uniforms and buffers setup (same as original)
    const u_projection = gl.getUniformLocation(program, "projection");
    const u_viewport = gl.getUniformLocation(program, "viewport");
    const u_focal = gl.getUniformLocation(program, "focal");
    const u_view = gl.getUniformLocation(program, "view");

    const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    var u_textureLocation = gl.getUniformLocation(program, "u_texture");
    gl.uniform1i(u_textureLocation, 0);

    const indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
    gl.vertexAttribDivisor(a_index, 1);

    // Camera and view setup functions (same as original)
    const resize = () => {
        gl.uniform2fv(u_focal, new Float32Array([camera.fx, camera.fy]));

        let w = config.W;
        let h = config.H;
        let ratio = Math.min(innerWidth / w, innerHeight / h);
        w = innerWidth / ratio;
        h = innerHeight / ratio;
        projectionMatrix = getProjectionMatrix(camera.fx, camera.fy, w, h);

        gl.uniform2fv(u_viewport, new Float32Array([w, h]));
        gl.canvas.width = Math.round(innerWidth / downsample);
        gl.canvas.height = Math.round(innerHeight / downsample);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.uniformMatrix4fv(u_projection, false, projectionMatrix);
    };

    window.addEventListener("resize", resize);
    resize();

    // Enhanced worker message handling with GoP support
    worker.onmessage = (e) => {
        if (e.data.buffer) {
            // Handle initial buffer setup
            splatData = new Uint8Array(e.data.buffer);
        } else if (e.data.texdata) {
            // Handle texture updates
            const { texdata, texwidth, texheight } = e.data;
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, texwidth, texheight, 0, 
                         gl.RGBA_INTEGER, gl.UNSIGNED_INT, texdata);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture);
        } else if (e.data.depthIndex) {
            // Handle depth sorting updates
            const { depthIndex, viewProj } = e.data;
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
            vertexCount = e.data.vertexCount;
        } else if (e.data.compressionStats) {
            // Handle compression statistics updates
            updateCompressionUI(e.data.compressionStats);
        }
    };

    // Input handling - Complete implementation
    let activeKeys = [];
    let currentCameraIndex = 0;

    window.addEventListener("keydown", (e) => {
        carousel = false;
        if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
        if (/\d/.test(e.key)) {
            currentCameraIndex = parseInt(e.key)
            camera = cameras[currentCameraIndex];
            viewMatrix = getViewMatrix(camera);
        }
        if (['-', '_'].includes(e.key)){
            currentCameraIndex = (currentCameraIndex + cameras.length - 1) % cameras.length;
            viewMatrix = getViewMatrix(cameras[currentCameraIndex]);
        }
        if (['+', '='].includes(e.key)){
            currentCameraIndex = (currentCameraIndex + 1) % cameras.length;
            viewMatrix = getViewMatrix(cameras[currentCameraIndex]);
        }
        camid.innerText = "cam  " + currentCameraIndex;
        if (e.code == "KeyV") {
            location.hash =
                "#" +
                JSON.stringify(
                    viewMatrix.map((k) => Math.round(k * 100) / 100),
                );
                camid.innerText =""
        } else if (e.code === "KeyP") {
            carousel = true;
            camid.innerText =""
        }
    });
    window.addEventListener("keyup", (e) => {
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });
    window.addEventListener("blur", () => {
        activeKeys = [];
    });

    window.addEventListener(
        "wheel",
        (e) => {
            carousel = false;
            e.preventDefault();
            const lineHeight = 10;
            const scale =
                e.deltaMode == 1
                    ? lineHeight
                    : e.deltaMode == 2
                    ? innerHeight
                    : 1;
            let inv = invert4(viewMatrix);
            if (e.shiftKey) {
                inv = translate4(
                    inv,
                    (e.deltaX * scale) / innerWidth,
                    (e.deltaY * scale) / innerHeight,
                    0,
                );
            } else if (e.ctrlKey || e.metaKey) {
                inv = translate4(
                    inv,
                    0,
                    0,
                    (-10 * (e.deltaY * scale)) / innerHeight,
                );
            } else {
                let d = 4;
                inv = translate4(inv, 0, 0, d);
                inv = rotate4(inv, -(e.deltaX * scale) / innerWidth, 0, 1, 0);
                inv = rotate4(inv, (e.deltaY * scale) / innerHeight, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);
            }

            viewMatrix = invert4(inv);
        },
        { passive: false },
    );

    let startX, startY, down;
    canvas.addEventListener("mousedown", (e) => {
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = e.ctrlKey || e.metaKey ? 2 : 1;
    });
    canvas.addEventListener("contextmenu", (e) => {
        carousel = false;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = 2;
    });

    canvas.addEventListener("mousemove", (e) => {
        e.preventDefault();
        if (down == 1) {
            let inv = invert4(viewMatrix);
            let dx = (5 * (e.clientX - startX)) / innerWidth;
            let dy = (5 * (e.clientY - startY)) / innerHeight;
            let d = 4;

            inv = translate4(inv, 0, 0, d);
            inv = rotate4(inv, dx, 0, 1, 0);
            inv = rotate4(inv, -dy, 1, 0, 0);
            inv = translate4(inv, 0, 0, -d);
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        } else if (down == 2) {
            let inv = invert4(viewMatrix);
            inv = translate4(
                inv,
                (-10 * (e.clientX - startX)) / innerWidth,
                0,
                (10 * (e.clientY - startY)) / innerHeight,
            );
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        }
    });
    canvas.addEventListener("mouseup", (e) => {
        e.preventDefault();
        down = false;
        startX = 0;
        startY = 0;
    });

    // Touch handling
    let altX = 0, altY = 0;
    canvas.addEventListener(
        "touchstart",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                carousel = false;
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                down = 1;
            } else if (e.touches.length === 2) {
                carousel = false;
                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
                down = 1;
            }
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchmove",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1 && down) {
                let inv = invert4(viewMatrix);
                let dx = (4 * (e.touches[0].clientX - startX)) / innerWidth;
                let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;

                let d = 4;
                inv = translate4(inv, 0, 0, d);
                inv = rotate4(inv, dx, 0, 1, 0);
                inv = rotate4(inv, -dy, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);

                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                const dtheta =
                    Math.atan2(startY - altY, startX - altX) -
                    Math.atan2(
                        e.touches[0].clientY - e.touches[1].clientY,
                        e.touches[0].clientX - e.touches[1].clientX,
                    );
                const dscale =
                    Math.hypot(startX - altX, startY - altY) /
                    Math.hypot(
                        e.touches[0].clientX - e.touches[1].clientX,
                        e.touches[0].clientY - e.touches[1].clientY,
                    );
                const dx =
                    (e.touches[0].clientX +
                        e.touches[1].clientX -
                        (startX + altX)) /
                    2;
                const dy =
                    (e.touches[0].clientY +
                        e.touches[1].clientY -
                        (startY + altY)) /
                    2;
                let inv = invert4(viewMatrix);
                inv = rotate4(inv, dtheta, 0, 0, 1);
                inv = translate4(inv, -dx / innerWidth, -dy / innerHeight, 0);
                inv = translate4(inv, 0, 0, 3 * (1 - dscale));
                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
            }
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchend",
        (e) => {
            e.preventDefault();
            down = false;
            startX = 0;
            startY = 0;
        },
        { passive: false },
    );

    // Gamepad handling
    window.addEventListener("gamepadconnected", (e) => {
        const gp = navigator.getGamepads()[e.gamepad.index];
        console.log(
            `Gamepad connected at index ${gp.index}: ${gp.id}. It has ${gp.buttons.length} buttons and ${gp.axes.length} axes.`,
        );
    });
    window.addEventListener("gamepaddisconnected", (e) => {
        console.log("Gamepad disconnected");
    });

    let leftGamepadTrigger, rightGamepadTrigger;

    let jumpDelta = 0;
    let vertexCount = 0;
    let lastFrame = 0;
    let avgFps = 0;
    let start = 0;
    let curFrame = 0;

    // Enhanced frame rendering loop
    const frame = (now) => {
        let inv = invert4(viewMatrix);
        let shiftKey = activeKeys.includes("Shift") || activeKeys.includes("ShiftLeft") || activeKeys.includes("ShiftRight")

        if (activeKeys.includes("ArrowUp")) {
            if (shiftKey) {
                inv = translate4(inv, 0, -0.03, 0);
            } else {
                inv = translate4(inv, 0, 0, 0.1);
            }
        }
        if (activeKeys.includes("ArrowDown")) {
            if (shiftKey) {
                inv = translate4(inv, 0, 0.03, 0);
            } else {
                inv = translate4(inv, 0, 0, -0.1);
            }
        }
        if (activeKeys.includes("ArrowLeft"))
            inv = translate4(inv, -0.03, 0, 0);
        if (activeKeys.includes("ArrowRight"))
            inv = translate4(inv, 0.03, 0, 0);
        if (activeKeys.includes("KeyA")) inv = rotate4(inv, -0.01, 0, 1, 0);
        if (activeKeys.includes("KeyD")) inv = rotate4(inv, 0.01, 0, 1, 0);
        if (activeKeys.includes("KeyQ")) inv = rotate4(inv, 0.01, 0, 0, 1);
        if (activeKeys.includes("KeyE")) inv = rotate4(inv, -0.01, 0, 0, 1);
        if (activeKeys.includes("KeyW")) inv = rotate4(inv, 0.005, 1, 0, 0);
        if (activeKeys.includes("KeyS")) inv = rotate4(inv, -0.005, 1, 0, 0);

        const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
        let isJumping = activeKeys.includes("Space");
        for (let gamepad of gamepads) {
            if (!gamepad) continue;

            const axisThreshold = 0.1;
            const moveSpeed = 0.06;
            const rotateSpeed = 0.02;

            if (Math.abs(gamepad.axes[0]) > axisThreshold) {
                inv = translate4(inv, moveSpeed * gamepad.axes[0], 0, 0);
                carousel = false;
            }
            if (Math.abs(gamepad.axes[1]) > axisThreshold) {
                inv = translate4(inv, 0, 0, -moveSpeed * gamepad.axes[1]);
                carousel = false;
            }
            if(gamepad.buttons[12].pressed || gamepad.buttons[13].pressed){
                inv = translate4(inv, 0, -moveSpeed*(gamepad.buttons[12].pressed - gamepad.buttons[13].pressed), 0);
                carousel = false;
            }

            if(gamepad.buttons[14].pressed || gamepad.buttons[15].pressed){
                inv = translate4(inv, -moveSpeed*(gamepad.buttons[14].pressed - gamepad.buttons[15].pressed), 0, 0);
                carousel = false;
            }

            if (Math.abs(gamepad.axes[2]) > axisThreshold) {
                inv = rotate4(inv, rotateSpeed * gamepad.axes[2], 0, 1, 0);
                carousel = false;
            }
            if (Math.abs(gamepad.axes[3]) > axisThreshold) {
                inv = rotate4(inv, -rotateSpeed * gamepad.axes[3], 1, 0, 0);
                carousel = false;
            }

            let tiltAxis = gamepad.buttons[6].value - gamepad.buttons[7].value;
            if (Math.abs(tiltAxis) > axisThreshold) {
                inv = rotate4(inv, rotateSpeed * tiltAxis, 0, 0, 1);
                carousel = false;
            }
            if (gamepad.buttons[4].pressed && !leftGamepadTrigger) {
                camera = cameras[(cameras.indexOf(camera)+1)%cameras.length]
                inv = invert4(getViewMatrix(camera));
                carousel = false;
            }
            if (gamepad.buttons[5].pressed && !rightGamepadTrigger) {
                camera = cameras[(cameras.indexOf(camera)+cameras.length-1)%cameras.length]
                inv = invert4(getViewMatrix(camera));
                carousel = false;
            }
            leftGamepadTrigger = gamepad.buttons[4].pressed;
            rightGamepadTrigger = gamepad.buttons[5].pressed;
            if (gamepad.buttons[0].pressed) {
                isJumping = true;
                carousel = false;
            }
            if(gamepad.buttons[3].pressed){
                carousel = true;
            }
        }

        if (
            ["KeyJ", "KeyK", "KeyL", "KeyI"].some((k) => activeKeys.includes(k))
        ) {
            let d = 4;
            inv = translate4(inv, 0, 0, d);
            inv = rotate4(
                inv,
                activeKeys.includes("KeyJ")
                    ? -0.05
                    : activeKeys.includes("KeyL")
                    ? 0.05
                    : 0,
                0,
                1,
                0,
            );
            inv = rotate4(
                inv,
                activeKeys.includes("KeyI")
                    ? 0.05
                    : activeKeys.includes("KeyK")
                    ? -0.05
                    : 0,
                1,
                0,
                0,
            );
            inv = translate4(inv, 0, 0, -d);
        }

        viewMatrix = invert4(inv);

        if (carousel) {
            let inv = invert4(defaultViewMatrix);
            const t = 0;
            viewMatrix = invert4(inv);
        }

        if (isJumping) {
            jumpDelta = Math.min(1, jumpDelta + 0.05);
        } else {
            jumpDelta = Math.max(0, jumpDelta - 0.05);
        }

        let inv2 = invert4(viewMatrix);
        inv2 = translate4(inv2, 0, -jumpDelta, 0);
        inv2 = rotate4(inv2, -0.1 * jumpDelta, 1, 0, 0);
        let actualViewMatrix = invert4(inv2);
        
        const viewProj = multiply4(projectionMatrix, actualViewMatrix || viewMatrix);
        worker.postMessage({ view: viewProj });

        const currentFps = 1000 / (now - lastFrame) || 0;
        avgFps = avgFps * 0.9 + currentFps * 0.1;

        if (vertexCount > 0) {
            document.getElementById("spinner").style.display = "none";
            gl.uniformMatrix4fv(u_view, false, actualViewMatrix || viewMatrix);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
        } else {
            gl.clear(gl.COLOR_BUFFER_BIT);
            document.getElementById("spinner").style.display = "";
            start = Date.now() + 2000;
        }

        fps.innerText = Math.round(avgFps);
        document.getElementById("frame").innerText = curFrame;
        
        lastFrame = now;
        requestAnimationFrame(frame);
    };
    frame();

    // Enhanced streaming pipeline with I/P frame support
    let bytesRead = 0;
    let gaussians = [];
    let rowBuffer = new Uint8Array(STREAM_ROW_LENGTH);
    let rowBufferOffset = 0;

    console.log("Enhanced GoP streaming started");

    // Step 1: Load initial I-frame (frame 0)
    console.log("Loading initial I-frame...");
    while (bytesRead < TOTAL_CAP * STREAM_ROW_LENGTH) {
        let { done, value } = await reader.read();
        
        // Handle partial reads
        let value_offset = 0;
        if (rowBufferOffset > 0) {
            if (value.length + rowBufferOffset < STREAM_ROW_LENGTH) {
                rowBuffer.set(value, rowBufferOffset);
                rowBufferOffset += value.length;
                continue;
            }
            value_offset = STREAM_ROW_LENGTH - rowBufferOffset;
            rowBuffer.set(value.slice(0, STREAM_ROW_LENGTH - rowBufferOffset), rowBufferOffset);
            gaussians.push(PARSE_RAW_BYTES_COMPACT(rowBuffer)[0]);
            bytesRead += STREAM_ROW_LENGTH;
            rowBuffer.fill(0);
            rowBufferOffset = 0;
        }
        
        // Batch parse
        value = value.slice(value_offset);
        const num_of_gs = Math.floor(value.length / STREAM_ROW_LENGTH);
        let parsed = PARSE_RAW_BYTES_COMPACT(value.slice(0, num_of_gs * STREAM_ROW_LENGTH));
        gaussians = gaussians.concat(parsed);
        let value_rest = value.slice(num_of_gs * STREAM_ROW_LENGTH);
        rowBuffer.set(value_rest);
        rowBufferOffset = value_rest.length;
        bytesRead += num_of_gs * STREAM_ROW_LENGTH;
    }
    
    console.log(`Initial I-frame loaded: ${gaussians.length} Gaussians`);
    
    // Store I-frame and send to worker
    storeIFrameBuffer(gaussians.slice(0, TOTAL_CAP), 0);
    let ret = GS_TO_VERTEX_COMPACT(gaussians.slice(0, TOTAL_CAP), true);
    worker.postMessage({
        buffer: ret.all.buffer,
        vertexCount: TOTAL_CAP,
        frameType: 'I',
        frameNumber: 0
    });

    // Step 2: Setup frame ticker with GoP support
    curFrame = 1;
    const FPS = config.FPS;
    let frameEvents = [];
    let loadedFrame = 0;
    
    const pauseBtn = document.getElementById("pause");
    var paused = 0;
    pauseBtn.addEventListener("click", () => {
        paused = !paused;
        pauseBtn.innerText = paused ? "⏩ Resume" : "⏸️ Pause";
    });

    // Enhanced frame ticker with I/P frame logic
    const frame_ticker = setInterval(() => {
        if (loadedFrame < Math.min(curFrame, MAX_FRAME - 2)) {
            console.log("Buffer underrun - loaded:", loadedFrame, "/ cur:", curFrame);
            return;
        }
        if (paused) return;
        
        let updated = false;
        let reset_slices = [];
        const currentFrameIsIFrame = isIFrame(curFrame);
        const currentGoPNum = getGoPNumber(curFrame);
        
        // Check for GoP boundary
        if (curFrame % GOP_SIZE === 0 && curFrame > 0) {
            console.log(`Starting new GoP ${currentGoPNum} at frame ${curFrame}`);
            // Keep I-frame buffer for the new GoP
        }
        
        // Process frame events
        for (let i = 0; i < frameEvents.length; i++) {
            const event = frameEvents[i];
            
            if (event.frame === curFrame) {
                if (event.type === "reset") {
                    reset_slices.push(event.sliceId);
                    worker.postMessage({
                        resetSlice: {
                            sliceId: event.sliceId,
                            data: event.data,
                            frame: event.frame,
                            isIFrame: event.isIFrame,
                            gopNumber: currentGoPNum
                        }
                    });
                    updated = true;
                    
                } else if (event.type === "append") {
                    worker.postMessage({
                        appendSlice: {
                            sliceId: event.sliceId,
                            data: event.data,
                            frame: event.frame,
                            isIFrame: event.isIFrame,
                            gopNumber: currentGoPNum
                        }
                    });
                    updated = true;
                }
            }
        }
        
        if (updated) {
            const frameType = currentFrameIsIFrame ? 'I' : 'P';
            console.log(`Frame #${curFrame} (${frameType}): processing`);
            
            worker.postMessage({
                reSort: {
                    reset_slices: reset_slices,
                    frameType: frameType,
                    frameNumber: curFrame,
                    gopNumber: currentGoPNum
                }
            });
            
            curFrame++;
            update_FPS();
            
            // Update compression stats display
            updateCompressionDisplay();
        }
        
        // Loop handling
        if (curFrame >= MAX_FRAME) {
            console.log("Restarting stream from I-frame");
            let ret = GS_TO_VERTEX_COMPACT(iFrameBuffer.slice(0, TOTAL_CAP), true);
            worker.postMessage({
                buffer: ret.all.buffer,
                vertexCount: TOTAL_CAP,
                frameType: 'I',
                frameNumber: 0
            });
            curFrame = 1;
        }
        
        update_curframe(curFrame);
    }, Math.ceil(1000 / FPS));

    // Step 3: Enhanced streaming loop with I/P frame parsing
    bytesRead = 0;
    let init_gs = gaussians.splice(0, TOTAL_CAP);
    loadedFrame = 0;
    
    console.log("Starting enhanced streaming loop...");
    
    while (loadedFrame < MAX_FRAME) {
        let { done, value } = await reader.read();
        if (done) break;
        
        // Handle partial reads (same logic as before)
        let value_offset = 0;
        if (rowBufferOffset > 0) {
            if (value.length + rowBufferOffset < STREAM_ROW_LENGTH) {
                rowBuffer.set(value, rowBufferOffset);
                rowBufferOffset += value.length;
                continue;
            }
            value_offset = STREAM_ROW_LENGTH - rowBufferOffset;
            rowBuffer.set(value.slice(0, STREAM_ROW_LENGTH - rowBufferOffset), rowBufferOffset);
            gaussians.push(PARSE_RAW_BYTES_COMPACT(rowBuffer)[0]);
            bytesRead += STREAM_ROW_LENGTH;
            rowBuffer.fill(0);
            rowBufferOffset = 0;
        }
        
        value = value.slice(value_offset);
        const num_of_gs = Math.floor(value.length / STREAM_ROW_LENGTH);
        let parsed = PARSE_RAW_BYTES_COMPACT(value.slice(0, num_of_gs * STREAM_ROW_LENGTH));
        gaussians = gaussians.concat(parsed);
        let value_rest = value.slice(num_of_gs * STREAM_ROW_LENGTH);
        rowBuffer.set(value_rest);
        rowBufferOffset = value_rest.length;
        bytesRead += num_of_gs * STREAM_ROW_LENGTH;
        
        // Process frames when enough data is available
        while (bytesRead >= SLICE_CAP * STREAM_ROW_LENGTH && loadedFrame < MAX_FRAME) {
            loadedFrame++;
            update_buffered(loadedFrame);
            
            const isCurrentIFrame = isIFrame(loadedFrame);
            const currentGoPNum = getGoPNumber(loadedFrame);
            
            if (isCurrentIFrame) {
                // I-Frame processing
                console.log(`Processing I-frame #${loadedFrame}`);
                let frameData = gaussians.splice(0, SLICE_CAP);
                storeIFrameBuffer(frameData, loadedFrame);
                
                let ret = GS_TO_VERTEX_COMPACT(frameData);
                console.log(`I-frame #${loadedFrame}: ${ret.spans.length} spans`);
                
                // Create frame events for I-frame
                let curFrameOverwrite = false;
                for (let span of ret.spans) {
                    frameEvents.push(createFrameEvent(
                        span.frame,
                        ret.all.slice(span.from * VERTEX_ROW_LENGTH, span.to * VERTEX_ROW_LENGTH),
                        (span.frame === loadedFrame) ? "reset" : "append",
                        true, // isIFrame
                        null,
                        null
                    ));
                    if (span.frame === loadedFrame) curFrameOverwrite = true;
                }
                
                if (!curFrameOverwrite) {
                    frameEvents.push(createFrameEvent(loadedFrame, new Uint8Array(0), "reset", true));
                }
                
            } else {
                // P-Frame processing
                console.log(`Processing P-frame #${loadedFrame}`);
                let compressedData = gaussians.splice(0, SLICE_CAP);
                
                if (COMPRESSION_ENABLED && iFrameBuffer) {
                    // Parse P-frame compressed data
                    let compressedBytes = new Uint8Array(SLICE_CAP * STREAM_ROW_LENGTH);
                    for (let i = 0; i < compressedData.length && i < SLICE_CAP; i++) {
                        // Convert Gaussian object back to binary format for parsing
                        // This is a simplified approach - in production, you'd receive binary P-frame data directly
                        let offset = i * STREAM_ROW_LENGTH;
                        // Write frame info
                        let view = new DataView(compressedBytes.buffer, offset, STREAM_ROW_LENGTH);
                        view.setUint16(0, compressedData[i].start_frame, true);
                        view.setUint16(2, compressedData[i].end_frame, true);
                        // Write position
                        view.setFloat32(4, compressedData[i].xyz[0], true);
                        view.setFloat32(8, compressedData[i].xyz[1], true);
                        view.setFloat32(12, compressedData[i].xyz[2], true);
                        // Write other attributes...
                    }
                    
                    let pFrameData = parsePFrameData(compressedBytes);
                    
                    if (pFrameData.isValid && iFrameBuffer) {
                        // Reconstruct P-frame
                        let reconstructedFrame = createReconstruction(
                            iFrameBuffer,
                            pFrameData.reusableIndices,
                            pFrameData.residualGaussians
                        );
                        
                        let ret = GS_TO_VERTEX_COMPACT(reconstructedFrame);
                        console.log(`P-frame #${loadedFrame}: reconstructed ${reconstructedFrame.length} Gaussians`);
                        
                        // Create frame events for P-frame
                        let curFrameOverwrite = false;
                        for (let span of ret.spans) {
                            frameEvents.push(createFrameEvent(
                                span.frame,
                                ret.all.slice(span.from * VERTEX_ROW_LENGTH, span.to * VERTEX_ROW_LENGTH),
                                (span.frame === loadedFrame) ? "reset" : "append",
                                false, // isIFrame
                                pFrameData.reusableIndices,
                                pFrameData.residualGaussians
                            ));
                            if (span.frame === loadedFrame) curFrameOverwrite = true;
                        }
                        
                        if (!curFrameOverwrite) {
                            frameEvents.push(createFrameEvent(loadedFrame, new Uint8Array(0), "reset", false));
                        }
                        
                    } else {
                        console.warn(`P-frame #${loadedFrame}: Invalid data or missing I-frame buffer`);
                        // Fallback to treating as regular frame
                        let ret = GS_TO_VERTEX_COMPACT(compressedData);
                        for (let span of ret.spans) {
                            frameEvents.push(createFrameEvent(span.frame, ret.all.slice(span.from * VERTEX_ROW_LENGTH, span.to * VERTEX_ROW_LENGTH), "reset", false));
                        }
                    }
                    
                } else {
                    // Compression disabled or no I-frame buffer - treat as regular frame
                    let ret = GS_TO_VERTEX_COMPACT(compressedData);
                    for (let span of ret.spans) {
                        frameEvents.push(createFrameEvent(span.frame, ret.all.slice(span.from * VERTEX_ROW_LENGTH, span.to * VERTEX_ROW_LENGTH), "reset", false));
                    }
                }
            }
            
            bytesRead -= SLICE_CAP * STREAM_ROW_LENGTH;
        }
    }
    
    console.log("Enhanced streaming pipeline completed");
}

/**
 * Enhanced worker with GoP support - Complete Implementation
 */
function createEnhancedWorker(self, SLICE_CAP, SLICE_NUM, GOP_SIZE) {
    var global_log_2 = {"sort":[], "texture":[]};
    var compressionStats = {
        totalFrames: 0,
        iFrames: 0,
        pFrames: 0,
        bandwidthSaved: 0
    };
    
    function average(arr) {
        return arr.reduce((a,b) => a+b, 0) / arr.length;
    }
    
    let buffer;
    let vertexCount = 0;
    let viewProj;
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    let lastProj = [];
    let depthIndex = new Uint32Array();
    let lastVertexCount = 0;
    let currentFrameType = 'I';
    let currentGoPNumber = 0;

    // Float conversion functions
    var _floatView = new Float32Array(1);
    var _int32View = new Int32Array(_floatView.buffer);

    function floatToHalf(float) {
        _floatView[0] = float;
        var f = _int32View[0];

        var sign = (f >> 31) & 0x0001;
        var exp = (f >> 23) & 0x00ff;
        var frac = f & 0x007fffff;

        var newExp;
        if (exp == 0) {
            newExp = 0;
        } else if (exp < 113) {
            newExp = 0;
            frac |= 0x00800000;
            frac = frac >> (113 - exp);
            if (frac & 0x01000000) {
                newExp = 1;
                frac = 0;
            }
        } else if (exp < 142) {
            newExp = exp - 112;
        } else {
            newExp = 31;
            frac = 0;
        }

        return (sign << 15) | (newExp << 10) | (frac >> 13);
    }

    function packHalf2x16(x, y) {
        return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
    }

    function generateTexture() {
        let start = Date.now();
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        const u_buffer = new Uint8Array(buffer);

        var texwidth = 1024 * 2;
        var texheight = Math.ceil((2 * vertexCount) / texwidth);
        var texdata = new Uint32Array(texwidth * texheight * 4);
        var texdata_c = new Uint8Array(texdata.buffer);
        var texdata_f = new Float32Array(texdata.buffer);

        for (let i = 0; i < vertexCount; i++) {
            // x, y, z
            texdata_f[8 * i + 0] = f_buffer[8 * i + 0];
            texdata_f[8 * i + 1] = f_buffer[8 * i + 1];
            texdata_f[8 * i + 2] = f_buffer[8 * i + 2];

            // r, g, b, a
            texdata_c[4 * (8 * i + 7) + 0] = u_buffer[32 * i + 24 + 0];
            texdata_c[4 * (8 * i + 7) + 1] = u_buffer[32 * i + 24 + 1];
            texdata_c[4 * (8 * i + 7) + 2] = u_buffer[32 * i + 24 + 2];
            texdata_c[4 * (8 * i + 7) + 3] = u_buffer[32 * i + 24 + 3];

            // quaternions
            let scale = [
                f_buffer[8 * i + 3 + 0],
                f_buffer[8 * i + 3 + 1],
                f_buffer[8 * i + 3 + 2],
            ];
            let rot = [
                (u_buffer[32 * i + 28 + 0] - 128) / 128,
                (u_buffer[32 * i + 28 + 1] - 128) / 128,
                (u_buffer[32 * i + 28 + 2] - 128) / 128,
                (u_buffer[32 * i + 28 + 3] - 128) / 128,
            ];

            // Compute the matrix product of S and R (M = S * R)
            const M = [
                1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
                2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
                2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),

                2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
                2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),

                2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
                2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
            ].map((k, i) => k * scale[Math.floor(i / 3)]);

            const sigma = [
                M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
                M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
                M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
                M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
                M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
                M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
            ];

            texdata[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
            texdata[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
            texdata[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);
        }
        let end = Date.now();
        global_log_2["texture"].push(end - start);
        console.log(`avg time cost of each step: 
            sort: ${average(global_log_2["sort"])}ms
            texture: ${average(global_log_2["texture"])}ms
            `);
        self.postMessage({ texdata, texwidth, texheight}, [texdata.buffer]);
    }

    function runSort(viewProj, enforce_update=false) {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        if (lastVertexCount == vertexCount && !enforce_update) {
            let dot =
                lastProj[2] * viewProj[2] +
                lastProj[6] * viewProj[6] +
                lastProj[10] * viewProj[10];
            if (Math.abs(dot - 1) < 0.01) {
                return;
            }
        } else {
            lastVertexCount = vertexCount;
        }
        let start = Date.now();
        console.time("sort");
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++) {
            let depth =
                ((viewProj[2] * f_buffer[8 * i + 0] +
                    viewProj[6] * f_buffer[8 * i + 1] +
                    viewProj[10] * f_buffer[8 * i + 2]) *
                    4096) |
                0;
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
        }

        // This is a 16 bit single-pass counting sort
        let depthInv = (256 * 256) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256 * 256);
        for (let i = 0; i < vertexCount; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++)
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        depthIndex = new Uint32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++)
            depthIndex[starts0[sizeList[i]]++] = i;

        console.timeEnd("sort");
        let end = Date.now();
        global_log_2["sort"].push(end - start);
        lastProj = viewProj;
        // put texture update and depth update together
        generateTexture();
        self.postMessage({ depthIndex, viewProj, vertexCount }, [
            depthIndex.buffer,
        ]);
    }

    function processPlyBuffer(inputBuffer) {
        const ubuf = new Uint8Array(inputBuffer);
        const header = new TextDecoder().decode(ubuf.slice(0, 1024 * 10));
        const header_end = "end_header\n";
        const header_end_index = header.indexOf(header_end);
        if (header_end_index < 0)
            throw new Error("Unable to read .ply file header");
        const vertexCount = parseInt(/element vertex (\d+)\n/.exec(header)[1]);
        console.log("Vertex Count", vertexCount);
        let row_offset = 0,
            offsets = {},
            types = {};
        const TYPE_MAP = {
            double: "getFloat64",
            int: "getInt32",
            uint: "getUint32",
            float: "getFloat32",
            short: "getInt16",
            ushort: "getUint16",
            uchar: "getUint8",
        };
        for (let prop of header
            .slice(0, header_end_index)
            .split("\n")
            .filter((k) => k.startsWith("property "))) {
            const [p, type, name] = prop.split(" ");
            const arrayType = TYPE_MAP[type] || "getInt8";
            types[name] = arrayType;
            offsets[name] = row_offset;
            row_offset += parseInt(arrayType.replace(/[^\d]/g, "")) / 8;
        }
        console.log("Bytes per row", row_offset, types, offsets);

        let dataView = new DataView(
            inputBuffer,
            header_end_index + header_end.length,
        );
        let row = 0;
        const attrs = new Proxy(
            {},
            {
                get(target, prop) {
                    if (!types[prop]) throw new Error(prop + " not found");
                    return dataView[types[prop]](
                        row * row_offset + offsets[prop],
                        true,
                    );
                },
            },
        );

        console.time("calculate importance");
        let sizeList = new Float32Array(vertexCount);
        let sizeIndex = new Uint32Array(vertexCount);
        for (row = 0; row < vertexCount; row++) {
            sizeIndex[row] = row;
            if (!types["scale_0"]) continue;
            const size =
                Math.exp(attrs.scale_0) *
                Math.exp(attrs.scale_1) *
                Math.exp(attrs.scale_2);
            const opacity = 1 / (1 + Math.exp(-attrs.opacity));
            sizeList[row] = size * opacity;
        }
        console.timeEnd("calculate importance");

        console.time("sort");
        sizeIndex.sort((b, a) => sizeList[a] - sizeList[b]);
        console.timeEnd("sort");

        const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
        const buffer = new ArrayBuffer(rowLength * vertexCount);

        console.time("build buffer");
        for (let j = 0; j < vertexCount; j++) {
            row = sizeIndex[j];

            const position = new Float32Array(buffer, j * rowLength, 3);
            const scales = new Float32Array(buffer, j * rowLength + 4 * 3, 3);
            const rgba = new Uint8ClampedArray(
                buffer,
                j * rowLength + 4 * 3 + 4 * 3,
                4,
            );
            const rot = new Uint8ClampedArray(
                buffer,
                j * rowLength + 4 * 3 + 4 * 3 + 4,
                4,
            );

            if (types["scale_0"]) {
                const qlen = Math.sqrt(
                    attrs.rot_0 ** 2 +
                        attrs.rot_1 ** 2 +
                        attrs.rot_2 ** 2 +
                        attrs.rot_3 ** 2,
                );

                rot[0] = (attrs.rot_0 / qlen) * 128 + 128;
                rot[1] = (attrs.rot_1 / qlen) * 128 + 128;
                rot[2] = (attrs.rot_2 / qlen) * 128 + 128;
                rot[3] = (attrs.rot_3 / qlen) * 128 + 128;

                scales[0] = Math.exp(attrs.scale_0);
                scales[1] = Math.exp(attrs.scale_1);
                scales[2] = Math.exp(attrs.scale_2);
            } else {
                scales[0] = 0.01;
                scales[1] = 0.01;
                scales[2] = 0.01;

                rot[0] = 255;
                rot[1] = 0;
                rot[2] = 0;
                rot[3] = 0;
            }

            position[0] = attrs.x;
            position[1] = attrs.y;
            position[2] = attrs.z;

            if (types["f_dc_0"]) {
                const SH_C0 = 0.28209479177387814;
                rgba[0] = (0.5 + SH_C0 * attrs.f_dc_0) * 255;
                rgba[1] = (0.5 + SH_C0 * attrs.f_dc_1) * 255;
                rgba[2] = (0.5 + SH_C0 * attrs.f_dc_2) * 255;
            } else {
                rgba[0] = attrs.red;
                rgba[1] = attrs.green;
                rgba[2] = attrs.blue;
            }
            if (types["opacity"]) {
                rgba[3] = (1 / (1 + Math.exp(-attrs.opacity))) * 255;
            } else {
                rgba[3] = 255;
            }
        }
        console.timeEnd("build buffer");
        return buffer;
    }

    const throttledSort = () => {
        if (!sortRunning) {
            sortRunning = true;
            let lastView = viewProj;
            runSort(lastView);
            setTimeout(() => {
                sortRunning = false;
                if (lastView !== viewProj) {
                    throttledSort();
                }
            }, 0);
        }
    };

    let sortRunning;
    let slicePtr = new Array(SLICE_NUM).fill(0);
    
    function getSlice(sId) {
        return new Uint8Array(buffer, sId * SLICE_CAP * rowLength, SLICE_CAP * rowLength);
    }

    // Enhanced message handling with GoP support
    self.onmessage = (e) => {
        if (e.data.ply) {
            vertexCount = 0;
            runSort(viewProj);
            buffer = processPlyBuffer(e.data.ply);
            vertexCount = Math.floor(buffer.byteLength / rowLength);
            postMessage({ buffer: buffer });
            
        } else if (e.data.buffer) {
            buffer = e.data.buffer;
            vertexCount = e.data.vertexCount;
            currentFrameType = e.data.frameType || 'I';
            
            if (currentFrameType === 'I') {
                compressionStats.iFrames++;
                console.log(`Worker: I-frame buffer set with ${vertexCount} vertices`);
            }
            
        } else if (e.data.vertexCount) {
            vertexCount = e.data.vertexCount;
            
        } else if (e.data.view) {
            viewProj = e.data.view;
            throttledSort();
            
        } else if (e.data.resetSlice) {
            let sId = e.data.resetSlice.sliceId;
            let data = e.data.resetSlice.data;
            let isIFrame = e.data.resetSlice.isIFrame || false;
            let gopNumber = e.data.resetSlice.gopNumber || 0;
            
            let num_of_gs = Math.floor(data.length / rowLength);
            let num_of_gs_capped = Math.min(num_of_gs, SLICE_CAP);
            let bufferSlice = getSlice(sId);
            
            bufferSlice.set(data.slice(0, num_of_gs_capped * rowLength));
            if (num_of_gs_capped < SLICE_CAP) {
                bufferSlice.fill(0, num_of_gs_capped * rowLength);
            }
            slicePtr[sId] = num_of_gs_capped;
            
            console.log(`Worker: Slice #${sId} reset with ${slicePtr[sId]} gaussians (${isIFrame ? 'I' : 'P'}-frame, GoP ${gopNumber})`);
            
        } else if (e.data.appendSlice) {
            let sId = e.data.appendSlice.sliceId;
            let data = e.data.appendSlice.data;
            let isIFrame = e.data.appendSlice.isIFrame || false;
            
            if (slicePtr[sId] >= SLICE_CAP) return;
            
            let num_of_gs = Math.floor(data.length / rowLength);
            let num_of_gs_capped = Math.min(num_of_gs, SLICE_CAP - slicePtr[sId]);
            
            if (num_of_gs > num_of_gs_capped) {
                console.warn(`Worker: Slice #${sId} overflow from frame #${e.data.appendSlice.frame}`);
            }
            
            let bufferSlice = getSlice(sId);
            bufferSlice.set(data.slice(0, num_of_gs_capped * rowLength), slicePtr[sId] * rowLength);
            slicePtr[sId] += num_of_gs_capped;
            
            console.log(`Worker: Slice #${sId} appended ${num_of_gs_capped} gaussians (${isIFrame ? 'I' : 'P'}-frame)`);
            
        } else if (e.data.reSort) {
            currentFrameType = e.data.reSort.frameType || 'I';
            currentGoPNumber = e.data.reSort.gopNumber || 0;
            
            if (currentFrameType === 'P') {
                compressionStats.pFrames++;
            }
            
            runSort(viewProj, true);
            
            self.postMessage({
                compressionStats: compressionStats
            });
        }
    };
}

/**
 * UI update functions for compression statistics
 */
function updateCompressionDisplay() {
    const statsElement = document.getElementById("compression-stats");
    if (statsElement) {
        const compressionRatio = compressionStats.totalFrames > 0 ? 
            (compressionStats.savedBandwidth / compressionStats.totalBandwidth * 100).toFixed(1) : 0;
        
        statsElement.innerHTML = `
            I-frames: ${compressionStats.iFrames} | 
            P-frames: ${compressionStats.pFrames} | 
            Compression: ${compressionRatio}%
        `;
    }
}

function updateCompressionUI(stats) {
    compressionStats = { ...compressionStats, ...stats };
    updateCompressionDisplay();
}

/**
 * Initialize enhanced streaming
 */
function initializeEnhancedStreaming(config) {
    console.log("Initializing Enhanced GoP Streaming...");
    return enhancedStreamingMain(config);
}

// Export functions for integration
window.enhancedStreaming = {
    initialize: initializeEnhancedStreaming,
    isIFrame: isIFrame,
    getGoPNumber: getGoPNumber,
    compressionStats: compressionStats
};