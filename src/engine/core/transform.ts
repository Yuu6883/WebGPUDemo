import { mat4, quat, ReadonlyVec3 } from 'gl-matrix';

const inverted = new Float32Array(16);

export default class Transform {
    // prettier-ignore
    private readonly buffer = new Float32Array([
        0, 0, 0, 0, 
        0, 0, 0, 1, 
        1, 1, 1, 0
    ]);

    public readonly model: Float32Array;

    constructor(model: Float32Array) {
        this.model = model;
    }

    get position() {
        return this.buffer.slice(0, 3);
    }

    set position(pos: ReadonlyVec3) {
        this.buffer[0] = pos[0];
        this.buffer[1] = pos[1];
        this.buffer[2] = pos[2];
    }

    get rotation() {
        return this.buffer.slice(4, 8);
    }

    set rotation(rot: quat) {
        quat.normalize(this.buffer.subarray(4, 8), rot);
    }

    get scale() {
        return this.buffer.slice(8, 11);
    }

    set scale(value: ReadonlyVec3) {
        this.buffer[8] = value[0];
        this.buffer[9] = value[1];
        this.buffer[10] = value[2];
    }

    update() {
        mat4.fromRotationTranslationScale(
            this.model.subarray(0, 16),
            this.buffer.subarray(4, 8),
            this.buffer.subarray(0, 3),
            this.buffer.subarray(8, 11),
        );
    }

    updateInverse() {
        mat4.invert(inverted, this.model.subarray(0, 16));
        mat4.transpose(this.model.subarray(16, 32), inverted);
    }
}
