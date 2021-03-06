import { clamp } from '../math/util';
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

export default class CameraRotator {
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
        this.cam.targetDistance *= 1 + delta / 1000;
    }

    onMouseMove(nx: number, ny: number) {
        const MaxDelta = 100;
        const dx = clamp(nx - this.mouseX, -MaxDelta, MaxDelta);
        const dy = clamp(-(ny - this.mouseY), -MaxDelta, MaxDelta);

        this.mouseX = nx;
        this.mouseY = ny;

        if (this.leftDown) {
            const rate = 1;
            this.cam.azimuth = this.cam.azimuth + dx * rate;
            this.cam.incline = clamp(this.cam.incline - dy * rate, -90, 90);
        }

        if (this.rightDown) {
            const rate = 0.005;
            this.cam.distance = clamp(this.cam.distance * (1 - dx * rate), 0.01, 1000);
        }
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
