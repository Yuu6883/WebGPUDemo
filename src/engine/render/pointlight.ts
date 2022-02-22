export default class PointLight {
    static readonly STRIDE = 8;

    readonly buffer = new Float32Array(PointLight.STRIDE);

    get position() {
        return this.buffer.slice(0, 3);
    }

    set position(value: Float32Array) {
        this.buffer.set(value.subarray(0, 3), 0);
    }

    get color() {
        return this.buffer.slice(4, 7);
    }

    set color(value: Float32Array) {
        this.buffer.set(value.subarray(0, 3), 4);
    }

    get radius() {
        return this.buffer[7];
    }

    set radius(value: number) {
        this.buffer[7] = value;
    }
}
