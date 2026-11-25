//
//  SamModels.swift
//  iOS_SAM_ViT
//
//  Created by Sasha Krigel on 11/15/26.
//

import CoreML

final class SAMModels {
    let encoder: SAMImageEncoder
    let decoder: SAMPromptDecoder

    init() async throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all            // let Core ML use the Neural Engine
        encoder = try await SAMImageEncoder.load(configuration: config)
        decoder = try await SAMPromptDecoder.load(configuration: config)
    }
}

/*
 
 Usage:
 // Encoder: input is a 512x512 image, ALREADY letterboxed (long side 512, top-left, pad RGB 124,116,104)
 let encIn = try SAMImageEncoderInput(imageWith: letterboxedCGImage)
 let emb = try await models.encoder.prediction(input: encIn).image_embeddings   // MLMultiArray 1x256x64x64, cache this

 // Decoder: points in the 1024 frame (same per-axis scaling as to_prompt_frame)
 let coords = try MLMultiArray(shape: [1, 2, 2], dataType: .float32)   // N = 2: click + (0,0) pad
 let labels = try MLMultiArray(shape: [1, 2], dataType: .float32)      // [1, -1]
 // ... fill coords/labels ...
 let out = try await models.decoder.prediction(image_embeddings: emb, point_coords: coords, point_labels: labels)
 // out.masks: 1x4x256x256 logits, out.iou_predictions: 1x4
 
 
 */

