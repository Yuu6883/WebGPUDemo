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
        color: number[];
        direction: number[];
    }[];
    ambient_light: number[];
}

export interface AsteroidWebAssemblyModule extends WebAssembly.Exports {
    _start();
    run_bench();
    init_map();
    set_asteroid_size(size: number);
    populate_asteroids();
    tick(vel: number);
    get_asteroid_size(): number;
    get_asteroid_state(): number;
    get_asteroid_pos_x(): number;
    get_asteroid_pos_y(): number;
}

declare global {
    interface Window {
        require: typeof require;
    }
}
