import { ReadonlyVec3, vec3 } from 'gl-matrix';
import { RenderPass } from '../render/interfaces';
import StaticMesh from '../render/staticmesh';

// prettier-ignore
const positions = new Float32Array(3 * 36);

// prettier-ignore
const normals = new Float32Array(3 * 36);

// prettier-ignore
const uvs = new Float32Array(2 * 36);

const p1 = new Float32Array(3);
const p2 = new Float32Array(3);
const cross = new Float32Array(3);
const norm = new Float32Array(3);

const initOneFace = (
    a: number,
    b: number,
    c: number,
    d: number,
    points: ReadonlyVec3[],
    offset: number,
) => {
    vec3.sub(p1, points[c], points[b]);
    vec3.sub(p2, points[a], points[b]);
    vec3.cross(cross, p1, p2);
    vec3.normalize(norm, cross);

    for (let i = 0; i < 6; i++) {
        const elem_offset = (offset + i) * 3;
        positions.set(points[[a, b, c, a, c, d][i]], elem_offset);
        normals.set(norm, elem_offset);
    }
};

const POINTS = [
    [-0.5, -0.5, +0.5],
    [-0.5, +0.5, +0.5],
    [+0.5, +0.5, +0.5],
    [+0.5, -0.5, +0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, +0.5, -0.5],
    [+0.5, +0.5, -0.5],
    [+0.5, -0.5, -0.5],
] as ReadonlyVec3[];

initOneFace(1, 0, 3, 2, POINTS, 0 * 6);
initOneFace(2, 3, 7, 6, POINTS, 1 * 6);
initOneFace(3, 0, 4, 7, POINTS, 2 * 6);
initOneFace(6, 5, 1, 2, POINTS, 3 * 6);
initOneFace(4, 5, 6, 7, POINTS, 4 * 6);
initOneFace(5, 4, 0, 1, POINTS, 5 * 6);

// let out: string[] = [];
// for (let i = 0; i < positions.length; i += 3) {
//     out.push(`vec3<f32>(${positions[i]}, ${positions[i + 1]}, ${positions[i + 2]})`);
// }
// console.log(out.join(',\n'));

export const CubePositions = positions;

export default class Cube extends StaticMesh {
    constructor(pass: RenderPass, extent: ReadonlyVec3 = [1, 1, 1]) {
        super(pass, positions, normals, uvs);
        // TODO: generate UVs and set scale
    }
}
