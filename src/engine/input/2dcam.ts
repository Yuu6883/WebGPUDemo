import { clamp } from '../math/util';
import { RESOLUTION } from '../render/base';
import Camera from '../render/camera';

enum InputState {
    MOUSE_UP,
    MOUSE_DOWN,
}

enum InputButtons {
    MOUSE_LEFT = 0,
    MOUSE_MIDDLE = 1,
    MOUSE_RIGHT = 2,
}

export default class Camera2DController {
    private leftDown: boolean;
    private middleDown: boolean;
    private rightDown: boolean;

    private mouseX: number;
    private mouseY: number;

    private readonly cam: Camera;

    constructor(camera: Camera) {
        this.cam = camera;

        window.addEventListener('contextmenu', e => e.preventDefault());
        window.addEventListener('mousemove', e => this.onMouseMove(e.clientX, e.clientY));
        window.addEventListener('mousedown', e => {
            if (e.target != document.body) return;
            this.onClick(e.button, InputState.MOUSE_DOWN, e.clientX, e.clientY);
        });
        window.addEventListener('mouseup', e => {
            if (e.target != document.body) return;
            this.onClick(e.button, InputState.MOUSE_UP, e.clientX, e.clientY);
        });
        window.addEventListener('wheel', e => this.onMouseWheel(e.deltaY));
    }

    onMouseWheel(delta: number) {
        this.cam.zoomTarget *= 1 - delta / 1000;
    }

    onMouseMove(clientX: number, clientY: number) {
        let dx = (2 * clientX) / window.innerWidth - 1;
        let dy = (2 * clientY) / window.innerHeight - 1;
        const r = RESOLUTION;
        const ratio = window.innerWidth / window.innerHeight / (r[0] / r[1]);
        if (ratio > 1) {
            dy /= ratio;
        } else {
            dx *= ratio;
        }

        if (this.rightDown) this.cam.pan(this.mouseX - dx, dy - this.mouseY);

        this.mouseX = dx;
        this.mouseY = dy;
    }

    public onClick(button: number, mode: InputState, x: number, y: number) {
        if (button == InputButtons.MOUSE_LEFT) {
            this.leftDown = mode == InputState.MOUSE_DOWN;
        } else if (button == InputButtons.MOUSE_MIDDLE) {
            this.middleDown = mode == InputState.MOUSE_DOWN;
        } else if (button == InputButtons.MOUSE_RIGHT) {
            this.rightDown = mode == InputState.MOUSE_DOWN;
        }
    }
}
