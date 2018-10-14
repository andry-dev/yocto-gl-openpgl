//
// LICENSE:
//
// Copyright (c) 2016 -- 2018 Fabio Pellacini
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include "../yocto/ygl.h"
#include "../yocto/yglio.h"
using namespace ygl;

int main(int argc, char* argv[]) {
    // trace options
    auto params = trace_params();

    // parse command line
    auto parser = make_cmdline_parser(
        argc, argv, "Offline path tracing", "ytrace");
    params.camid       = parse_arg(parser, "--camera", 0, "Camera index.");
    params.yresolution = parse_arg(
        parser, "--resolution,-r", 512, "Image vertical resolution.");
    params.nsamples = parse_arg(
        parser, "--nsamples,-s", 256, "Number of samples.");
    params.tracer = (trace_type)parse_arge(
        parser, "--tracer,-t", 0, "Trace type.", trace_type_names);
    params.nbounces = parse_arg(
        parser, "--nbounces", 8, "Maximum number of bounces.");
    params.pixel_clamp = parse_arg(
        parser, "--pixel-clamp", 100.0f, "Final pixel clamping.");
    params.noparallel = parse_arg(
        parser, "--noparallel", false, "Disable parallel execution.");
    params.seed = parse_arg(
        parser, "--seed", 13, "Seed for the random number generators.");
    params.nbatch = parse_arg(parser, "--nbatch,-b", 16, "Samples per batch.");
    auto save_batch = parse_arg(
        parser, "--save-batch", false, "Save images progressively");
    auto exposure = parse_arg(parser, "--exposure,-e", 0.0f, "Hdr exposure");
    auto gamma    = parse_arg(parser, "--gamma,-g", 2.2f, "Hdr gamma");
    auto filmic   = parse_arg(parser, "--filmic", false, "Hdr filmic");
    auto embree   = parse_arg(parser, "--embree", false, "Use Embree ratracer");
    auto double_sided = parse_arg(
        parser, "--double-sided,-D", false, "Double-sided rendering.");
    auto add_skyenv = parse_arg(
        parser, "--add-skyenv,-E", false, "add missing env map");
    auto imfilename = parse_arg(
        parser, "--output-image,-o", "out.hdr", "Image filename");
    auto filename = parse_arg(
        parser, "scene", "scene.json", "Scene filename", true);
    check_cmdline(parser);

    // scene loading
    log_info("loading scene {}", filename);
    auto load_start = get_time();
    auto scn        = load_scene(filename);
    if (!scn) log_fatal("cannot load scene " + filename);
    log_info("loading in {}", format_duration(get_time() - load_start).c_str());

    // tesselate
    log_info("tesselating scene elements");
    tesselate_subdivs(scn);

    // add components
    log_info("adding scene elements");
    if (add_skyenv && scn->environments.empty()) {
        scn->environments.push_back(make_sky_environment("sky"));
        scn->textures.push_back(scn->environments.back()->ke_txt);
    }
    if (double_sided)
        for (auto mat : scn->materials) mat->double_sided = true;
    for (auto& err : validate(scn)) log_error("warning: {}", err);

    // build bvh
    log_info("building bvh");
    auto bvh_start = get_time();
    auto bvh       = build_bvh(scn, true, embree);
    log_info(
        "building bvh in {}", format_duration(get_time() - bvh_start).c_str());

    // init renderer
    log_info("initializing lights");
    auto lights = make_trace_lights(scn, params);

    // initialize rendering objects
    log_info("initializing tracer data");
    auto state = make_trace_state(scn, params);

    // render
    log_info("rendering image");
    auto render_start = get_time();
    auto done         = false;
    while (!done) {
        log_info("rendering sample {}/{}", state->sample, params.nsamples);
        auto block_start = get_time();
        done             = trace_samples(state, scn, bvh, lights, params);
        log_info("rendering block in {}",
            format_duration(get_time() - block_start).c_str());
        if (save_batch) {
            auto filename = replace_extension(imfilename,
                to_string(state->sample) + "." + get_extension(imfilename));
            log_info("saving image {}", filename.c_str());
            if (!save_tonemapped_image(
                    filename, state->img, exposure, gamma, filmic))
                log_fatal("cannot save image " + filename);
        }
    }
    log_info("rendering image in {}",
        format_duration(get_time() - render_start).c_str());

    // save image
    log_info("saving image {}", imfilename.c_str());
    if (!save_tonemapped_image(imfilename, state->img, exposure, gamma, filmic))
        log_fatal("cannot save image " + imfilename);

    // cleanup
    delete state;
    delete scn;
    delete bvh;
    delete lights;

    // done
    return 0;
}
