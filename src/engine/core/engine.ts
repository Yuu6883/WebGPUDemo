import Renderer from '../render/base';
import { EngineParam } from '../types';

export default class Engine {
    private renderer: Renderer;
    public readonly params: EngineParam;

    constructor(params: EngineParam) {
        this.params = params;
        this.renderer = new Renderer(this);
    }

    init() {
        this.renderer.init();
    }
}
