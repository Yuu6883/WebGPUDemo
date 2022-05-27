import { mat4, ReadonlyVec3, vec3, glMatrix } from 'gl-matrix';
import Scene from './scene';

export default class Camera {
    private scene: Scene;

    public fov = 0;
    public aspect = 0;
    public nearClip = 0;
    public farClip = 0;

    public distance = 0;
    public azimuth = 0;
    public incline = 0;

    public targetDistance = 0;

    public readonly view = new Float32Array(16 + 4);

    constructor(scene: Scene) {
        this.scene = scene;
        this.reset();
    }

    public update() {
        this.distance += (this.targetDistance - this.distance) / 60;

        const temp = mat4.create();
        const world = mat4.create();
        world[14] = this.distance;

        const rotX = mat4.fromYRotation(mat4.create(), glMatrix.toRadian(-this.azimuth));
        const rotY = mat4.fromXRotation(mat4.create(), glMatrix.toRadian(-this.incline));
        mat4.mul(temp, rotX, rotY);

        const final = mat4.mul(mat4.create(), temp, world);

        mat4.getTranslation(this.view.subarray(16), final);

        const view = mat4.invert(mat4.create(), final);

        const proj = mat4.perspective(
            mat4.create(),
            glMatrix.toRadian(this.fov),
            this.aspect,
            this.nearClip,
            this.farClip,
        );

        mat4.mul(this.view, proj, view);
    }

    public reset() {
        this.fov = 60;
        this.aspect = 1.33;
        this.nearClip = 0.1;
        this.farClip = 2500;

        this.targetDistance = this.distance = 100;
        this.azimuth = 0;
        this.incline = 20;
    }
}

const UP = new Float32Array([0, 1, 0]);

const temp = new Float32Array(3);

export class OldCamera {
    private scene: Scene;

    private _pov = (2 * Math.PI) / 5;
    private readonly _pos = new Float32Array(3);
    private readonly _dir = new Float32Array(3);

    private readonly projection = new Float32Array(16);
    private readonly view = new Float32Array(16);
    public readonly vp = new Float32Array(16);

    constructor(scene: Scene) {
        this.scene = scene;

        mat4.perspectiveZO(
            this.projection,
            this._pov,
            scene.renderer.aspectRatio,
            1,
            2000,
        );
        this._pos.set([100, 100, 100]);
        this.lookAt([0, 0, 0]);
    }

    updateMatrix() {
        vec3.add(temp, this._pos, this._dir);
        mat4.lookAt(this.view, this._pos, temp, UP);
        mat4.mul(this.vp, this.projection, this.view);
    }

    lookAt(target: ReadonlyVec3) {
        vec3.sub(temp, target, this._pos);
        vec3.normalize(this._dir, temp);
        this.updateMatrix();
    }

    get position() {
        return this._pos.slice();
    }

    set position(target: ReadonlyVec3) {
        this._pos.set(target);
        this.updateMatrix();
    }
}
