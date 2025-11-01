import { ReadonlyVec2, ReadonlyVec3 } from 'gl-matrix';

export default class SpriteParam {
    public static readonly BYTES = 80;

    private readonly view: DataView;
    constructor(view: DataView) {
        this.view = view;
    }

    get size(): ReadonlyVec2 {
        return [this.view.getFloat32(0, true), this.view.getFloat32(4, true)];
    }

    set size(value: ReadonlyVec2) {
        this.view.setFloat32(0, value[0], true);
        this.view.setFloat32(4, value[1], true);
    }

    get uv_scale(): ReadonlyVec2 {
        return [this.view.getFloat32(8, true), this.view.getFloat32(12, true)];
    }

    set uv_scale(value: ReadonlyVec2) {
        this.view.setFloat32(8, value[0], true);
        this.view.setFloat32(12, value[1], true);
    }

    get variant_offset(): number {
        return this.view.getUint32(16, true);
    }

    set variant_offset(value: number) {
        this.view.setUint32(16, value, true);
    }

    get variant_count(): number {
        return this.view.getUint32(20, true);
    }

    set variant_count(value: number) {
        this.view.setUint32(20, value, true);
    }

    get normal_strength(): number {
        return this.view.getFloat32(24, true);
    }

    set normal_strength(value: number) {
        this.view.setFloat32(24, value, true);
    }

    get light_width(): number {
        return this.view.getFloat32(28, true);
    }

    set light_width(value: number) {
        this.view.setFloat32(28, value, true);
    }

    get brightness(): number {
        return this.view.getFloat32(32, true);
    }

    set brightness(value: number) {
        this.view.setFloat32(32, value, true);
    }

    get specular_strength(): number {
        return this.view.getFloat32(36, true);
    }

    set specular_strength(value: number) {
        this.view.setFloat32(36, value, true);
    }

    get specular_power(): number {
        return this.view.getFloat32(40, true);
    }

    set specular_power(value: number) {
        this.view.setFloat32(40, value, true);
    }

    get specular_purity(): number {
        return this.view.getFloat32(44, true);
    }

    set specular_purity(value: number) {
        this.view.setFloat32(44, value, true);
    }

    get sss_contrast(): number {
        return this.view.getFloat32(48, true);
    }

    set sss_contrast(value: number) {
        this.view.setFloat32(48, value, true);
    }

    get sss_amount(): number {
        return this.view.getFloat32(52, true);
    }

    set sss_amount(value: number) {
        this.view.setFloat32(52, value, true);
    }

    get light_offset(): number {
        return this.view.getUint32(56, true);
    }

    set light_offset(value: number) {
        this.view.setUint32(56, value, true);
    }

    get light_count(): number {
        return this.view.getUint32(60, true);
    }

    set light_count(value: number) {
        this.view.setUint32(60, value, true);
    }

    get ambient_light(): ReadonlyVec3 {
        return [
            this.view.getFloat32(64, true),
            this.view.getFloat32(68, true),
            this.view.getFloat32(72, true),
        ];
    }

    set ambient_light(value: ReadonlyVec3) {
        this.view.setFloat32(64, value[0], true);
        this.view.setFloat32(68, value[1], true);
        this.view.setFloat32(72, value[2], true);
    }

    get flags(): number {
        return this.view.getUint32(76, true);
    }

    set flags(value: number) {
        this.view.setUint32(76, value, true);
    }
}
