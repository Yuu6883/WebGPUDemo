export interface EngineParam {
    canvas: HTMLCanvasElement | OffscreenCanvas;
    screen: {
        width: number;
        height: number;
    };
    dpr: number;
}

declare global {
    interface Window {
        require: typeof require;
    }
}
