import Renderer from './base';

export default class Scene {
    public renderer: Renderer;

    constructor(renderer: Renderer) {
        this.renderer = renderer;
    }
}
