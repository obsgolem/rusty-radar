use bevy::{
    prelude::*,
    reflect::TypeUuid,
    render::{
        mesh::{self, shape, VertexAttributeValues},
        pipeline::{PipelineDescriptor, PrimitiveTopology::TriangleList, RenderPipeline},
        render_graph::{base, AssetRenderResourcesNode, RenderGraph},
        renderer::RenderResources,
        shader::{ShaderStage, ShaderStages},
    },
};
use bevy_orbit_controls::{OrbitCamera, OrbitCameraPlugin};

use crate::{
    geodesic_polyhedron::generate_polyhedron,
    helper::{decibels, decibels_or_else},
    helper_traits::{SphericalFunction, SphericalFunctionHelper},
};

#[derive(RenderResources, Default, TypeUuid)]
#[uuid = "0320b9b8-b3a3-4baa-8bfa-c94008177b17"]
struct VertexColorMaterial;

const VERTEX_SHADER: &str = r#"
#version 450
layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec3 Vertex_Color;
layout(location = 0) out vec3 v_color;
layout(set = 0, binding = 0) uniform CameraViewProj {
    mat4 ViewProj;
};
layout(set = 1, binding = 0) uniform Transform {
    mat4 Model;
};
void main() {
    gl_Position = ViewProj * Model * vec4(Vertex_Position, 1.0);
    v_color = Vertex_Color;
}
"#;

const FRAGMENT_SHADER: &str = r#"
#version 450
layout(location = 0) out vec4 o_Target;
layout(location = 0) in vec3 v_color;
void main() {
    o_Target = vec4(v_color, 1.0);
}
"#;

fn spherical_function_mesh(spherical_fn: &impl SphericalFunction, sample_density: usize) -> Mesh {
    let (mut verts_base, indices_base) = generate_polyhedron(sample_density);
    let mut items = spherical_fn.lookup_many(verts_base.iter().cloned());

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for x in items.iter_mut() {
        *x = decibels_or_else(*x, 0.1);
        if *x < min {
            min = *x;
        }
        if *x > max {
            max = *x;
        }
    }

    for vert in verts_base.iter_mut().enumerate() {
        *vert.1 *= ((items[vert.0] - min) / (max - min)).max(0.1) / vert.1.magnitude();
    }

    let color: Vec<_> = items
        .iter()
        .copied()
        .map(|x| [1. - (x - min) / (max - min), (x - min) / (max - min), 0.])
        .collect();

    let verts: Vec<_> = verts_base.iter().map(|x| [x[0], x[1], x[2]]).collect();

    let indices = indices_base
        .iter()
        .flat_map(|x| std::array::IntoIter::new([x.0 as u32, x.1 as u32, x.2 as u32]))
        .collect();

    let mut mesh = Mesh::new(TriangleList);
    mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, verts);
    mesh.set_attribute("Vertex_Color", color);
    mesh.set_indices(Some(mesh::Indices::U32(indices)));
    mesh
}

fn setup<T: SphericalFunction + Send + Sync + 'static>(
    spherical_fn: Res<T>,
    mut commands: Commands,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<VertexColorMaterial>>,
    mut render_graph: ResMut<RenderGraph>,
) {
    // Create a new shader pipeline
    let pipeline_handle = pipelines.add(PipelineDescriptor::default_config(ShaderStages {
        vertex: shaders.add(Shader::from_glsl(ShaderStage::Vertex, VERTEX_SHADER)),
        fragment: Some(shaders.add(Shader::from_glsl(ShaderStage::Fragment, FRAGMENT_SHADER))),
    }));

    // Add an AssetRenderResourcesNode to our Render Graph. This will bind
    // VertexColorMaterial resources to our shader
    render_graph.add_system_node(
        "vertex_color_material",
        AssetRenderResourcesNode::<VertexColorMaterial>::new(true),
    );

    // Add a Render Graph edge connecting our new "my_material" node to the main pass node. This
    // ensures "my_material" runs before the main pass
    render_graph
        .add_node_edge("vertex_color_material", base::node::MAIN_PASS)
        .unwrap();

    // Create a new material
    let material = materials.add(VertexColorMaterial {});

    let mesh = spherical_function_mesh(&*spherical_fn, 40);

    commands
        .spawn_bundle(MeshBundle {
            mesh: meshes.add(mesh),
            render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(
                pipeline_handle,
            )]),
            transform: Transform::from_matrix(Mat4::from_cols(
                Vec4::new(1., 0., 0., 0.),
                Vec4::new(0., 0., 1., 0.),
                Vec4::new(0., 1., 0., 0.),
                Vec4::new(0., 0., 0., 1.),
            )),
            ..Default::default()
        })
        .insert(material);
    // camera
    commands
        .spawn_bundle(PerspectiveCameraBundle {
            transform: Transform::from_xyz(2., 0.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        })
        .insert(OrbitCamera::default());
}

pub fn run<T: SphericalFunction + Sync + Send + 'static>(spherical_fn: T) {
    App::build()
        .add_plugins(DefaultPlugins)
        .add_plugin(OrbitCameraPlugin)
        .add_asset::<VertexColorMaterial>()
        .insert_resource(spherical_fn)
        .add_startup_system(setup::<T>.system())
        .run();
}
