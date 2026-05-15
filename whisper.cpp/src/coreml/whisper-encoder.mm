#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "whisper-encoder.h"

#import <CoreML/CoreML.h>

#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>

#if __cplusplus
extern "C" {
#endif

struct whisper_coreml_context {
    MLModel  * model;
    NSString * input_name;
    NSString * output_name;
};

static uint16_t whisper_f32_to_f16_bits(float value) {
    __fp16 half = (__fp16) value;
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static float whisper_f16_bits_to_f32(uint16_t bits) {
    __fp16 half;
    memcpy(&half, &bits, sizeof(half));
    return (float) half;
}

static NSString * whisper_coreml_pick_feature_name(NSDictionary<NSString *, MLFeatureDescription *> * features, NSArray<NSString *> * candidates) {
    for (NSString * candidate in candidates) {
        if ([features objectForKey:candidate] != nil) {
            return candidate;
        }
    }

    return features.allKeys.firstObject;
}

struct whisper_coreml_context * whisper_coreml_init(const char * path_model) {
    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];

    NSURL * url_model = [NSURL fileURLWithPath: path_model_str];

    // select which device to run the Core ML model on
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    // config.computeUnits = MLComputeUnitsCPUAndGPU;
    //config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    config.computeUnits = MLComputeUnitsAll;

    NSError * error = nil;
    MLModel * model = [MLModel modelWithContentsOfURL:url_model configuration:config error:&error];

    if (model == nil) {
        NSLog(@"whisper.cpp CoreML: failed to load model at %@: %@", path_model_str, error);
        return NULL;
    }

    whisper_coreml_context * ctx = new whisper_coreml_context;

    ctx->model = model;
    ctx->input_name = whisper_coreml_pick_feature_name(
        model.modelDescription.inputDescriptionsByName,
        @[@"melspectrogram_features", @"logmel_data"]
    );
    ctx->output_name = whisper_coreml_pick_feature_name(
        model.modelDescription.outputDescriptionsByName,
        @[@"encoder_output_embeds", @"output"]
    );

    if (ctx->input_name == nil || ctx->output_name == nil) {
        NSLog(@"whisper.cpp CoreML: model is missing usable input/output features");
        delete ctx;
        return NULL;
    }

    NSLog(@"whisper.cpp CoreML: input=%@ output=%@", ctx->input_name, ctx->output_name);

    return ctx;
}

void whisper_coreml_free(struct whisper_coreml_context * ctx) {
    delete ctx;
}

void whisper_coreml_encode(
        const whisper_coreml_context * ctx,
                             int64_t   n_ctx,
                             int64_t   n_mel,
                               float * mel,
                               float * out) {
    @autoreleasepool {
        MLFeatureDescription * input_desc = [ctx->model.modelDescription.inputDescriptionsByName objectForKey:ctx->input_name];
        MLMultiArrayDataType input_type = input_desc.multiArrayConstraint.dataType;
        NSArray<NSNumber *> * input_shape = input_desc.multiArrayConstraint.shape;
        const bool input_rank4 = input_shape.count == 4;

        std::vector<uint16_t> mel_f16;
        std::vector<float> mel_f32_padded;
        MLMultiArray * inMultiArray = nil;
        NSError * error = nil;
        const int64_t model_n_mel = input_shape.count > 1 ? input_shape[1].longLongValue : n_mel;
        const int64_t model_n_ctx = input_rank4 ? input_shape[3].longLongValue : input_shape[2].longLongValue;
        const int64_t input_n_mel = model_n_mel > 0 ? model_n_mel : n_mel;
        const int64_t input_n_ctx = model_n_ctx > 0 ? model_n_ctx : n_ctx;
        NSArray<NSNumber *> * mel_shape = input_rank4 ? @[@1, @(input_n_mel), @1, @(input_n_ctx)] : @[@1, @(input_n_mel), @(input_n_ctx)];
        NSArray<NSNumber *> * mel_strides = input_rank4 ? @[@(input_n_ctx*input_n_mel), @(input_n_ctx), @(input_n_ctx), @1] : @[@(input_n_ctx*input_n_mel), @(input_n_ctx), @1];

        if (input_type == MLMultiArrayDataTypeFloat16) {
            mel_f16.resize(input_n_ctx*input_n_mel);
            for (int64_t im = 0; im < n_mel && im < input_n_mel; ++im) {
                for (int64_t ic = 0; ic < n_ctx && ic < input_n_ctx; ++ic) {
                    mel_f16[im*input_n_ctx + ic] = whisper_f32_to_f16_bits(mel[im*n_ctx + ic]);
                }
            }

            inMultiArray = [
                [MLMultiArray alloc] initWithDataPointer: mel_f16.data()
                                                   shape: mel_shape
                                                dataType: MLMultiArrayDataTypeFloat16
                                                 strides: mel_strides
                                             deallocator: nil
                                                   error: &error
            ];
        } else {
            if (input_n_ctx != n_ctx || input_n_mel != n_mel) {
                mel_f32_padded.resize(input_n_ctx*input_n_mel);
                for (int64_t im = 0; im < n_mel && im < input_n_mel; ++im) {
                    memcpy(mel_f32_padded.data() + im*input_n_ctx, mel + im*n_ctx, sizeof(float)*std::min(n_ctx, input_n_ctx));
                }
                inMultiArray = [
                    [MLMultiArray alloc] initWithDataPointer: mel_f32_padded.data()
                                                       shape: mel_shape
                                                    dataType: MLMultiArrayDataTypeFloat32
                                                     strides: mel_strides
                                                 deallocator: nil
                                                       error: &error
                ];
            } else {
                inMultiArray = [
                    [MLMultiArray alloc] initWithDataPointer: mel
                                                       shape: mel_shape
                                                    dataType: MLMultiArrayDataTypeFloat32
                                                     strides: mel_strides
                                                 deallocator: nil
                                                       error: &error
                ];
            }
        }

        if (inMultiArray == nil) {
            NSLog(@"whisper.cpp CoreML: failed to create input multiarray: %@", error);
            return;
        }

        MLDictionaryFeatureProvider * inProvider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{ ctx->input_name: inMultiArray }
                         error:&error
        ];

        if (inProvider == nil) {
            NSLog(@"whisper.cpp CoreML: failed to create input provider: %@", error);
            return;
        }

        id<MLFeatureProvider> outProvider = [ctx->model predictionFromFeatures:inProvider error:&error];

        if (outProvider == nil) {
            NSLog(@"whisper.cpp CoreML: prediction failed: %@", error);
            return;
        }

        MLFeatureValue * outValue = [outProvider featureValueForName:ctx->output_name];
        MLMultiArray * outMultiArray = outValue.multiArrayValue;

        if (outMultiArray == nil) {
            NSLog(@"whisper.cpp CoreML: missing output multiarray %@", ctx->output_name);
            return;
        }

        NSArray<NSNumber *> * shape = outMultiArray.shape;
        NSArray<NSNumber *> * strides = outMultiArray.strides;
        const int rank = (int) shape.count;
        const int64_t requested_out_ctx = n_ctx/2;
        const int64_t model_out_ctx = input_n_ctx/2;
        int64_t n_state = outMultiArray.count / model_out_ctx;
        int64_t n_out_ctx = model_out_ctx;
        bool output_time_major = false;

        if (rank == 4) {
            n_state = shape[1].longLongValue;
            n_out_ctx = shape[3].longLongValue;
        } else if (rank == 3) {
            if (shape[1].longLongValue == model_out_ctx) {
                n_out_ctx = shape[1].longLongValue;
                n_state = shape[2].longLongValue;
                output_time_major = true;
            } else {
                n_state = shape[1].longLongValue;
                n_out_ctx = shape[2].longLongValue;
            }
        }

        const bool output_f16 = outMultiArray.dataType == MLMultiArrayDataTypeFloat16;
        const uint16_t * out_f16 = (const uint16_t *) outMultiArray.dataPointer;
        const float * out_f32 = (const float *) outMultiArray.dataPointer;

        n_out_ctx = std::min(n_out_ctx, requested_out_ctx);

        for (int64_t t = 0; t < n_out_ctx; ++t) {
            for (int64_t s = 0; s < n_state; ++s) {
                int64_t offset = s + t*n_state;

                if (rank == 4) {
                    offset =
                        0*strides[0].longLongValue +
                        s*strides[1].longLongValue +
                        0*strides[2].longLongValue +
                        t*strides[3].longLongValue;
                } else if (rank == 3) {
                    if (output_time_major) {
                        offset =
                            0*strides[0].longLongValue +
                            t*strides[1].longLongValue +
                            s*strides[2].longLongValue;
                    } else {
                        offset =
                            0*strides[0].longLongValue +
                            s*strides[1].longLongValue +
                            t*strides[2].longLongValue;
                    }
                }

                out[s + t*n_state] = output_f16 ? whisper_f16_bits_to_f32(out_f16[offset]) : out_f32[offset];
            }
        }
    }
}

#if __cplusplus
}
#endif
