import { ReadonlyVec3 } from 'gl-matrix';

export interface EngineParam {
    canvas: HTMLCanvasElement;
    screen: {
        width: number;
        height: number;
    };
    dpr: number;
}
export interface EntityRenderParams {
    name: string;
    variant: number;
    normal_strength: number;
    light_width: number;
    brightness: number;
    specular_strength: number;
    specular_power: number;
    specular_purity: number;
    sss_contrast: number;
    sss_amount: number;
    lights: {
        color: ReadonlyVec3;
        direction: ReadonlyVec3;
    }[];
    ambient_light: ReadonlyVec3;
    flat?: boolean;
}

export interface AsteroidWebAssemblyModule extends WebAssembly.Exports {
    _start();
    run_bench();
    init_map();
    set_asteroid_size(size: number);
    tick(vel: number);
    get_asteroid_size(): number;
    get_asteroid_state(): number;
    get_asteroid_pos_x(): number;
    get_asteroid_pos_y(): number;
    brush(x: number, y: number, radius: number, method: number, value: boolean);
    fill_asteroids(upper_bound: number);
    update_rng(
        x_offset: number,
        y_offset: number,
        x_range: number,
        y_range: number,
        vel: number,
    );
}

declare global {
    interface Window {
        require: typeof require;
    }
}
