import { mat4, ReadonlyVec3, vec3 } from 'gl-matrix';
import Scene from './scene';

const UP = new Float32Array([0, 1, 0]);

const temp = new Float32Array(3);

export default class Camera {
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
        this._pos.set([10, 10, 10]);
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
