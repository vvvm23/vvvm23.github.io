import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

class Cheem {
  constructor(scene) {
    this.scene = scene;
    this.rotv = Array.from({length: 3}, () => Math.random() * 1.0);
  }

  update_rotation(dt) {
    this.scene.rotation.x += dt * this.rotv[0];
    this.scene.rotation.y += dt * this.rotv[1];
    this.scene.rotation.z += dt * this.rotv[2];
  }
}

async function main() {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

  const renderer = new THREE.WebGLRenderer();
  renderer.setClearColor( 0x000000, 1 );
  renderer.setSize( window.innerWidth, window.innerHeight );
  document.body.appendChild( renderer.domElement );

  camera.position.z = 0;
  camera.lookAt(scene.position)

  const listener = new THREE.AudioListener();
  camera.add(listener);
  
  const music = new THREE.Audio(listener);

  const musicLoader = new THREE.AudioLoader();
  musicLoader.load('resources/music.ogg', function(buffer) {
    music.setBuffer(buffer);
    music.setLoop(true);
    music.setVolume(0.2);
    // music.play();
  });

  function play() {
    music.play()
  }

  document.body.addEventListener('click', play, true);

  const amb_light = new THREE.AmbientLight(0x404040);
  scene.add(amb_light);

  const loader = new GLTFLoader();
  let cheems = await loader.loadAsync('resources/cheems/scene.gltf', 
    function ( gltf ) {
      return gltf;
  }, undefined,
    function ( error ) {
      console.error( error );
    }
  );

  const cheems_scale = 0.05;
  cheems.scene.scale.set(cheems_scale, cheems_scale, cheems_scale);

  const box = new THREE.Box3().setFromObject(cheems.scene);
  cheems.scene.position.x = -(box.max.x + box.min.x) / 2;
  cheems.scene.position.y = -(box.max.y + box.min.y) / 2;
  cheems.scene.position.z = -(box.max.z + box.min.z) / 2;

  const pivot = new THREE.Object3D();
  pivot.add(cheems.scene);

  let pivots = [new Cheem(pivot)];
  for (var i = 1; i < 100; i++) {
    const rand_pos = Array.from({length: 3}, () => Math.random() * 40.0 - 20.0);
    const rand_rot = Array.from({length: 3}, () => Math.random() * 2 * Math.PI);
    const rand_scale = Math.random() * 2.0;
    const cloned_pivot = pivot.clone();
    cloned_pivot.position.set(...rand_pos);
    cloned_pivot.rotation.set(...rand_rot);
    cloned_pivot.scale.set(rand_scale, rand_scale, rand_scale);
    scene.add(cloned_pivot);

    if (i % 10 == 0) {
      const rand_col = Math.floor(Math.random()*16777215)
      const point_light = new THREE.PointLight(rand_col, 10, 20);
      point_light.position.set(...rand_pos);
      scene.add(point_light)
    }
    pivots.push(new Cheem(cloned_pivot));
  }

  const clock = new THREE.Clock();
  const rotate_speed = 0.1;

  const animate = function () {
    requestAnimationFrame( animate );
    const dt = clock.getDelta();

    pivots.forEach(function (pivot) {
      pivot.update_rotation(dt);
    })

    camera.rotation.z += dt * 0.1;
    camera.rotation.y -= dt * 0.1;
    camera.rotation.x -= dt * 0.1;

    renderer.render( scene, camera );
  };
  animate();
};

main();
