import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class SceneManager {
    constructor(container) {
        this.container = container;

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.gridHelper = null;
        this.axesHelper = null;

        this.init();
    }

    init() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111111);

        this.camera = new THREE.PerspectiveCamera(
            60,
            this.container.clientWidth / this.container.clientHeight,
            0.01,
            10000
        );
        this.camera.position.set(1.5, 1.5, 1.5);

        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            powerPreference: 'high-performance'
        });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.screenSpacePanning = true;
        this.controls.minDistance = 0.001;
        this.controls.maxDistance = 5000;
        this.controls.target.set(0, 0, 0);

        const ambient = new THREE.AmbientLight(0xffffff, 0.9);
        this.scene.add(ambient);

        const dir = new THREE.DirectionalLight(0xffffff, 0.6);
        dir.position.set(5, 10, 7);
        this.scene.add(dir);

        this.gridHelper = new THREE.GridHelper(2, 20, 0x555555, 0x333333);
        this.scene.add(this.gridHelper);

        this.axesHelper = new THREE.AxesHelper(0.5);
        this.scene.add(this.axesHelper);
    }

    setBackground(mode) {
        if (mode === 'light') {
            this.scene.background = new THREE.Color(0xf5f5f5);
        } else if (mode === 'blue') {
            this.scene.background = new THREE.Color(0x07111f);
        } else {
            this.scene.background = new THREE.Color(0x111111);
        }
    }

    toggleGrid() {
        this.gridHelper.visible = !this.gridHelper.visible;
    }

    toggleAxes() {
        this.axesHelper.visible = !this.axesHelper.visible;
    }

    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    render() {
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}