/*
 * Copyright (c) 2021-present Facebook, Inc.
 *
 * This file is based on vf_ssim.c from FFmpeg, which is licensed under the
 * terms of the GNU Lesser General Public license (LGPL).
 */

/*
 * Copyright (c) 2003-2013 Loren Merritt
 * Copyright (c) 2015 Paul B Mahol
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <essim.h>
#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "drawutils.h"
#include "formats.h"
#include "framesync.h"
#include "internal.h"
#include "video.h"

typedef struct ESSIMContext {
    const AVClass *class;
    FFFrameSync fs;
    FILE *stats_file;
    char *stats_file_str;
    int mode;
    int essim_mink_value;
    int pooling;
    int window_size;
    int window_stride;
    int d2h;
    int nb_components;
    int nb_threads;
    uint64_t nb_frames;
    double ssim[4], ssim_total;
    double essim[4], essim_total;
    char comps[4];
    double coefs[4];
    uint8_t rgba_map[4];
    int planewidth[4];
    int planeheight[4];
    int is_rgb;
    double **score;
    int (*ssim_plane)(AVFilterContext *ctx, void *arg,
                      int jobnr, int nb_jobs);
    //ESSIMDSPContext dsp;
    int bitdepth_minus_8;
    SSIM_CTX_ARRAY* essim_ctx_array[4];
} ESSIMContext;

#define OFFSET(x) offsetof(ESSIMContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption essim_options[] = {
    { "stats_file", "file where to store per-frame difference information", OFFSET(stats_file_str), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS },

    { "mode",  "mode of operation to trade off between performance and cross-platform precision", OFFSET(mode), AV_OPT_TYPE_INT,   { .i64 = SSIM_MODE_PERF_INT }, 0, 2, FLAGS, "mode" },
    { "ref",   "reference integer implementation with few optimizations",                         0,            AV_OPT_TYPE_CONST, { .i64 = SSIM_MODE_REF }, INT_MIN, INT_MAX, FLAGS, "mode" },
    { "int",   "optimized integer implementation, bit-exact to reference",                        0,            AV_OPT_TYPE_CONST, { .i64 = SSIM_MODE_PERF_INT }, INT_MIN, INT_MAX, FLAGS, "mode" },
    { "float", "optimized float implementation, not bit-exact to reference",                      0,            AV_OPT_TYPE_CONST, { .i64 = SSIM_MODE_PERF_FLOAT }, INT_MIN, INT_MAX, FLAGS, "mode" },

    { "mink",  "minkowski pooling coefficient", OFFSET(essim_mink_value), AV_OPT_TYPE_INT,   { .i64 = 3 }, 3, 4, FLAGS },

    { "pooling", "pooling mode for combining the scores across the frame ", OFFSET(pooling), AV_OPT_TYPE_INT,   { .i64 = SSIM_SPATIAL_POOLING_BOTH }, 0, 2, FLAGS, "pool"},
    { "mean", "arithmetic mean pooling",                                    0,               AV_OPT_TYPE_CONST, { .i64 = SSIM_SPATIAL_POOLING_AVERAGE }, INT_MIN, INT_MAX, FLAGS, "pool" },
    { "mink", "minkowksi pooling",                                          0,               AV_OPT_TYPE_CONST, { .i64 = SSIM_SPATIAL_POOLING_MINK }, INT_MIN, INT_MAX, FLAGS, "pool" },
    { "both", "both mean and mink pooling",                                 0,               AV_OPT_TYPE_CONST, { .i64 = SSIM_SPATIAL_POOLING_BOTH }, INT_MIN, INT_MAX, FLAGS, "pool" },

    { "window", "size of window (8 or 16)", OFFSET(window_size), AV_OPT_TYPE_INT, { .i64 = 8 }, 8, 16, FLAGS },
    { "stride", "window stride (4 or 8)",   OFFSET(window_stride), AV_OPT_TYPE_INT, { .i64 = 4 }, 4, 8, FLAGS },
    { "d2h", "ratio of viewing distance to viewport height", OFFSET(d2h), AV_OPT_TYPE_INT, { .i64 = 3 }, 1, 8, FLAGS },
    { NULL }
};

FRAMESYNC_DEFINE_CLASS(essim, ESSIMContext, fs);

static void set_meta(AVDictionary **metadata, const char *key, char comp, float d)
{
    char value[128];
    snprintf(value, sizeof(value), "%f", d);
    if (comp) {
        char key2[128];
        snprintf(key2, sizeof(key2), "%s%c", key, comp);
        av_dict_set(metadata, key2, value, 0);
    } else {
        av_dict_set(metadata, key, value, 0);
    }
}

#define SUM_LEN(w) (((w) >> 2) + 3)

typedef struct ThreadData {
    const uint8_t *main_data[4];
    const uint8_t *ref_data[4];
    int main_linesize[4];
    int ref_linesize[4];
    int planewidth[4];
    int planeheight[4];
    double **score;
    int nb_components;
    int bitdepth_minus_8;
    SSIM_CTX_ARRAY* essim_ctx_array[4];
} ThreadData;

static int essim_plane_16bit(AVFilterContext *ctx, void *arg,
                            int jobnr, int nb_jobs)
{
    ThreadData *td = arg;
    eSSIMResult res = SSIM_OK;
    ESSIMContext *s = ctx->priv;

    for (int c = 0; c < td->nb_components; c++) {
        int height = td->planeheight[c];
        const uint8_t* ref = td->main_data[c];
        const ptrdiff_t refStride = td->main_linesize[c];
        const uint8_t* cmp = td->ref_data[c];
        const ptrdiff_t cmpStride = td->ref_linesize[c];

        const int slice_start = ((height) * jobnr) / nb_jobs;
        const int slice_end = ((height) * (jobnr+1)) / nb_jobs;
        int roiY = slice_start;
        int roiHeight = slice_end - slice_start;

        SSIM_CTX_ARRAY* essim_ctx_arr = td->essim_ctx_array[c];
        SSIM_CTX* essim_ctx = ssim_access_ctx(essim_ctx_arr, jobnr);

        if (essim_ctx) {
            ssim_reset_ctx(essim_ctx);
            res = ssim_compute_ctx(essim_ctx, ref, refStride, cmp, cmpStride, roiY, roiHeight,
                                   s->essim_mink_value);
        } else {
            res = SSIM_ERR_FAILED;
        }
    }

    return (int)res;
}

static int essim_plane(AVFilterContext *ctx, void *arg,
                      int jobnr, int nb_jobs)
{
    ThreadData *td = arg;
    eSSIMResult res = SSIM_OK;
    ESSIMContext *s = ctx->priv;

    for (int c = 0; c < td->nb_components; c++) {

        int height = td->planeheight[c];
        const uint8_t* ref = td->main_data[c];
        const ptrdiff_t refStride = td->main_linesize[c];
        const uint8_t* cmp = td->ref_data[c];
        const ptrdiff_t cmpStride = td->ref_linesize[c];


        const int slice_start = ((height) * jobnr) / nb_jobs;
        const int slice_end = ((height) * (jobnr+1)) / nb_jobs;
        int roiY = slice_start;
        int roiHeight = slice_end - slice_start;

        SSIM_CTX_ARRAY* essim_ctx_arr = td->essim_ctx_array[c];
        SSIM_CTX* essim_ctx = ssim_access_ctx(essim_ctx_arr, jobnr);

        if (essim_ctx) {
            ssim_reset_ctx(essim_ctx);
            res = ssim_compute_ctx(essim_ctx, ref, refStride, cmp, cmpStride, roiY, roiHeight,
                                   s->essim_mink_value);
        } else {
            res = SSIM_ERR_FAILED;
        }
    }

    return (int)res;
}

static double essim_db(double ssim, double weight)
{
    return (fabs(weight - ssim) > 1e-9) ? 10.0 * log10(weight / (weight - ssim)) : INFINITY;
}

static int do_essim(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    ESSIMContext *s = ctx->priv;
    AVFrame *master, *ref;
    AVDictionary **metadata;
    double cSsim[4] = {0}, ssimv = 0.0;
    double cEssim[4] = {0}, essimv = 0.0;
    ThreadData td;
    int ret, i;

    ret = ff_framesync_dualinput_get(fs, &master, &ref);
    if (ret < 0)
        return ret;
    if (ctx->is_disabled || !ref)
        return ff_filter_frame(ctx->outputs[0], master);
    metadata = &master->metadata;

    s->nb_frames++;

    td.nb_components = s->nb_components;
    td.score = s->score;

    for (int n = 0; n < s->nb_components; n++) {
        td.main_data[n] = master->data[n];
        td.ref_data[n] = ref->data[n];
        td.main_linesize[n] = master->linesize[n];
        td.ref_linesize[n] = ref->linesize[n];
        td.planewidth[n] = s->planewidth[n];
        td.planeheight[n] = s->planeheight[n];
        td.essim_ctx_array[n] = s->essim_ctx_array[n];
    }

    ctx->internal->execute(ctx, s->ssim_plane, &td, NULL, FFMIN((s->planeheight[1] + 3) >> 2, s->nb_threads));

    for (i = 0; i < s->nb_components; i++) {
            float ssim_score = 0, essim_score = 0;
            float* pSsimScore = &ssim_score;
            float* pEssimScore = &essim_score;
            ssim_aggregate_score(pSsimScore, pEssimScore, s->essim_ctx_array[i],
                                 s->essim_mink_value);
            cSsim[i] = *pSsimScore;
            cEssim[i] = *pEssimScore;
    }

    for (i = 0; i < s->nb_components; i++) {
        ssimv += s->coefs[i] * cSsim[i];
        s->ssim[i] += cSsim[i];

        essimv += s->coefs[i] * cEssim[i];
        s->essim[i] += cEssim[i];
    }

    for (i = 0; i < s->nb_components; i++) {
        int cidx = s->is_rgb ? s->rgba_map[i] : i;
        if (s->pooling != SSIM_SPATIAL_POOLING_MINK ) {
            set_meta(metadata, "lavfi.ssim.", s->comps[i], cSsim[cidx]);
        }
        if (s->pooling != SSIM_SPATIAL_POOLING_AVERAGE ) {
            set_meta(metadata, "lavfi.essim.", s->comps[i], cEssim[cidx]);
        }
    }

    if (s->pooling != SSIM_SPATIAL_POOLING_MINK ) {
        s->ssim_total += ssimv;
        set_meta(metadata, "lavfi.ssim.All", 0, ssimv);
        set_meta(metadata, "lavfi.ssim.dB", 0, essim_db(ssimv, 1.0));
    }

    if (s->pooling != SSIM_SPATIAL_POOLING_AVERAGE ) {
        s->essim_total += essimv;
        set_meta(metadata, "lavfi.essim.All", 0, essimv);
        set_meta(metadata, "lavfi.essim.dB", 0, essim_db(essimv, 1.0));
    }

    if (s->stats_file) {
        if (s->pooling != SSIM_SPATIAL_POOLING_MINK ) {
            fprintf(s->stats_file, "SSIM  n:%"
            PRId64
            " ", s->nb_frames);
            for (i = 0; i < s->nb_components; i++) {
                int cidx = s->is_rgb ? s->rgba_map[i] : i;
                fprintf(s->stats_file, "%c:%f ", s->comps[i], cSsim[cidx]);
            }
            fprintf(s->stats_file, "All:%f (%f)\n", ssimv, essim_db(ssimv, 1.0));
        }

        if (s->pooling != SSIM_SPATIAL_POOLING_AVERAGE ) {
            fprintf(s->stats_file, "ESSIM n:%"
            PRId64
            " ", s->nb_frames);
            for (i = 0; i < s->nb_components; i++) {
                int cidx = s->is_rgb ? s->rgba_map[i] : i;
                fprintf(s->stats_file, "%c:%f ", s->comps[i], cEssim[cidx]);
            }
            fprintf(s->stats_file, "All:%f (%f)\n", essimv, essim_db(essimv, 1.0));
        }
    }

    return ff_filter_frame(ctx->outputs[0], master);
}

static av_cold int init(AVFilterContext *ctx)
{
    ESSIMContext *s = ctx->priv;

    if (s->stats_file_str) {
        if (!strcmp(s->stats_file_str, "-")) {
            s->stats_file = stdout;
        } else {
            s->stats_file = fopen(s->stats_file_str, "w");
            if (!s->stats_file) {
                int err = AVERROR(errno);
                char buf[128];
                av_strerror(err, buf, sizeof(buf));
                av_log(ctx, AV_LOG_ERROR, "Could not open stats file %s: %s\n",
                       s->stats_file_str, buf);
                return err;
            }
        }
    }

    s->fs.on_event = do_essim;
    return 0;
}

static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAY9, AV_PIX_FMT_GRAY10,
        AV_PIX_FMT_GRAY12, AV_PIX_FMT_GRAY14, AV_PIX_FMT_GRAY16,
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P,
        AV_PIX_FMT_YUV440P, AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUV410P,
        AV_PIX_FMT_YUVJ411P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ422P,
        AV_PIX_FMT_YUVJ440P, AV_PIX_FMT_YUVJ444P,
        AV_PIX_FMT_GBRP,
#define PF(suf) AV_PIX_FMT_YUV420##suf,  AV_PIX_FMT_YUV422##suf,  AV_PIX_FMT_YUV444##suf, AV_PIX_FMT_GBR##suf
        PF(P9), PF(P10), PF(P12), PF(P14), PF(P16),
        AV_PIX_FMT_NONE
};

static int config_input_ref(AVFilterLink *inlink)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    AVFilterContext *ctx  = inlink->dst;
    ESSIMContext *s = ctx->priv;
    int sum = 0, i;

	if(s->mode == SSIM_MODE_REF) {
		/*In ESSIM lib, ref mode does not support multi threading*/
		s->nb_threads = 1;
	} else {
		s->nb_threads = ff_filter_get_nb_threads(ctx);
	}
    s->nb_components = desc->nb_components;

    if (ctx->inputs[0]->w != ctx->inputs[1]->w ||
        ctx->inputs[0]->h != ctx->inputs[1]->h) {
        av_log(ctx, AV_LOG_ERROR, "Width and height of input videos must be same.\n");
        return AVERROR(EINVAL);
    }
    if (ctx->inputs[0]->format != ctx->inputs[1]->format) {
        av_log(ctx, AV_LOG_ERROR, "Inputs must be of same pixel format.\n");
        return AVERROR(EINVAL);
    }

    s->is_rgb = ff_fill_rgba_map(s->rgba_map, inlink->format) >= 0;
    s->comps[0] = s->is_rgb ? 'R' : 'Y';
    s->comps[1] = s->is_rgb ? 'G' : 'U';
    s->comps[2] = s->is_rgb ? 'B' : 'V';
    s->comps[3] = 'A';

    s->planeheight[1] = s->planeheight[2] = AV_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
    s->planeheight[0] = s->planeheight[3] = inlink->h;
    s->planewidth[1]  = s->planewidth[2]  = AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
    s->planewidth[0]  = s->planewidth[3]  = inlink->w;
    for (i = 0; i < s->nb_components; i++)
        sum += s->planeheight[i] * s->planewidth[i];
    for (i = 0; i < s->nb_components; i++)
        s->coefs[i] = (double) s->planeheight[i] * s->planewidth[i] / sum;

    s->bitdepth_minus_8 = desc->comp[0].depth - 8;
    s->ssim_plane = desc->comp[0].depth > 8 ? essim_plane_16bit : essim_plane;
    s->score = av_calloc(s->nb_threads, sizeof(*s->score));

    uint32_t windowSize = (uint32_t) s->window_size;
    uint32_t windowStride = (uint32_t) s->window_stride;
    uint32_t d2h = (uint32_t) s->d2h;
    eSSIMMode mode = s->mode;
    eSSIMFlags flags = s->pooling;


    for(int i = 0; i < s->nb_components; ++i){
        s->essim_ctx_array[i] = ssim_allocate_ctx_array(
            s->nb_threads,
            s->planewidth[i],
            s->planeheight[i],
            s->bitdepth_minus_8,
            desc->comp[0].depth > 8 ? SSIM_DATA_16BIT : SSIM_DATA_8BIT,

            windowSize,
            windowStride,
            d2h,
            mode,
            flags,
            s->essim_mink_value);
    }

    if (!s->score)
        return AVERROR(ENOMEM);

    for (int t = 0; t < s->nb_threads; t++) {
        s->score[t] = av_calloc(s->nb_components, sizeof(*s->score[0]));
        if (!s->score[t])
            return AVERROR(ENOMEM);
    }

    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    ESSIMContext *s = ctx->priv;
    AVFilterLink *mainlink = ctx->inputs[0];
    int ret;

    ret = ff_framesync_init_dualinput(&s->fs, ctx);
    if (ret < 0)
        return ret;
    outlink->w = mainlink->w;
    outlink->h = mainlink->h;
    outlink->time_base = mainlink->time_base;
    outlink->sample_aspect_ratio = mainlink->sample_aspect_ratio;
    outlink->frame_rate = mainlink->frame_rate;

    if ((ret = ff_framesync_configure(&s->fs)) < 0)
        return ret;

    outlink->time_base = s->fs.time_base;

    if (av_cmp_q(mainlink->time_base, outlink->time_base) ||
        av_cmp_q(ctx->inputs[1]->time_base, outlink->time_base))
        av_log(ctx, AV_LOG_WARNING, "not matching timebases found between first input: %d/%d and second input %d/%d, results may be incorrect!\n",
               mainlink->time_base.num, mainlink->time_base.den,
               ctx->inputs[1]->time_base.num, ctx->inputs[1]->time_base.den);

    return 0;
}

static int activate(AVFilterContext *ctx)
{
    ESSIMContext *s = ctx->priv;
    return ff_framesync_activate(&s->fs);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    ESSIMContext *s = ctx->priv;

    av_log(ctx, AV_LOG_INFO, "window_size: %d window_stride: %d d/h: %d \n",
            (int)s->window_size, (int)s->window_stride, (int)s->d2h);

    if (s->nb_frames > 0) {
        char buf[256];
        int i;


        if (s->pooling != SSIM_SPATIAL_POOLING_MINK ) {
            buf[0] = 0;
            for (i = 0; i < s->nb_components; i++) {
                int c = s->is_rgb ? s->rgba_map[i] : i;
                av_strlcatf(buf, sizeof(buf), " %c:%f (%f)", s->comps[i], s->ssim[c] / s->nb_frames,
                            essim_db(s->ssim[c], s->nb_frames));
            }

            av_log(ctx, AV_LOG_INFO,
                   "SSIM%s All:%f (%f) \n",
                   buf, (s->ssim_total / s->nb_frames), essim_db(s->ssim_total, s->nb_frames));
        }

        if (s->pooling != SSIM_SPATIAL_POOLING_AVERAGE ) {
            //Reset buffer
            buf[0] = 0;
            for (i = 0; i < s->nb_components; i++) {
                int c = s->is_rgb ? s->rgba_map[i] : i;
                av_strlcatf(buf, sizeof(buf), " %c:%f (%f)", s->comps[i], s->essim[c] / s->nb_frames,
                            essim_db(s->essim[c], s->nb_frames));

            }
            av_log(ctx, AV_LOG_INFO,
                   "ESSIM%s All:%f (%f) \n",
                   buf, (s->essim_total / s->nb_frames), essim_db(s->essim_total, s->nb_frames));
        }
    }

    ff_framesync_uninit(&s->fs);

    if (s->stats_file && s->stats_file != stdout)
        fclose(s->stats_file);

    for (int t = 0; t < s->nb_threads && s->score; t++)
        av_freep(&s->score[t]);
    av_freep(&s->score);

    for(int i = 0; i < s->nb_components; ++i){
        ssim_free_ctx_array(s->essim_ctx_array[i]);
    }
}

static const AVFilterPad essim_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
    },{
        .name         = "reference",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input_ref,
    },
};

static const AVFilterPad essim_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
};

AVFilter ff_vf_essim = {
    .name          = "essim",
    .description   = NULL_IF_CONFIG_SMALL("Calculate the eSSIM between two video streams."),
    .preinit       = essim_framesync_preinit,
    .init          = init,
    .uninit        = uninit,
    .activate      = activate,
    .priv_size     = sizeof(ESSIMContext),
    .priv_class    = &essim_class,
    FILTER_INPUTS(essim_inputs),
    FILTER_OUTPUTS(essim_outputs),
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL |
                     AVFILTER_FLAG_SLICE_THREADS             |
                     AVFILTER_FLAG_METADATA_ONLY,
};
