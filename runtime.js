const EPSILON = 1e-5;

export class NanoGPTTracer {
  constructor(bundle, tokenizer) {
    this.bundle = bundle;
    this.manifest = bundle.manifest;
    this.tensorMap = bundle.tensorMap;
    this.tokenizer = tokenizer;
    this.model = this.manifest.model;
  }

  step(promptText, options = {}) {
    const temperature = Math.max(0.01, Number(options.temperature ?? 1.0));
    const topK = Math.max(0, Number(options.topK ?? 0));
    const strategy = options.strategy ?? "argmax";
    const seed = Number(options.seed ?? 1337);

    let inputIds = this.tokenizer.encode(promptText);
    if (inputIds.length === 0) {
      const eosId = this.tokenizer.specialIds?.["<eos>"];
      if (eosId !== undefined) {
        inputIds = [eosId];
      } else {
        throw new Error("Prompt is empty.");
      }
    }

    const blockSize = this.model.block_size;
    const cropped = inputIds.length > blockSize;
    const contextIds = cropped ? inputIds.slice(-blockSize) : inputIds.slice();
    const positions = Int32Array.from({ length: contextIds.length }, (_, index) => index);
    const tokensText = contextIds.map((id) => this.tokenizer.tokenLabel(id));
    const startTime = performance.now();

    const trace = {
      sections: [],
      nodes: [],
      nextTokenId: null,
      nextTokenText: "",
      outputIds: [],
      outputText: "",
      cropped,
      temperature,
      topK,
      strategy,
      elapsedMs: 0,
    };

    const inputNode = this.#makeNode(
      "input.tokens",
      "Input Tokens",
      "Encoded context entering the transformer",
      "rgba(121, 240, 217, 0.18)",
      [
        makeTensor("token_ids", Int32Array.from(contextIds), [contextIds.length], { semantic: "tokens" }),
        makeTensor("positions", positions, [positions.length], { semantic: "vector" }),
      ],
      [
        `${contextIds.length}/${blockSize} tokens`,
        cropped ? "left-cropped to block size" : "full prompt preserved",
      ],
      { tokenIds: contextIds, tokenTexts: tokensText }
    );
    this.#pushSection(trace, "Input", `${contextIds.length} tokens`, [inputNode]);

    const wte = this.#tensor("transformer.wte.weight");
    const wpe = this.#tensor("transformer.wpe.weight");
    const tokEmb = embeddingLookup(wte.data, wte.shape[1], contextIds);
    const posEmb = embeddingLookup(wpe.data, wpe.shape[1], positions);
    let x = addSameShape(tokEmb, posEmb);

    const embeddingNode = this.#makeNode(
      "input.embeddings",
      "Embedding Sum",
      "Token embeddings plus position embeddings",
      "rgba(146, 167, 255, 0.18)",
      [
        makeTensor("token_embeddings", tokEmb, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
        makeTensor("position_embeddings", posEmb, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
        makeTensor("residual_stream_0", x, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
      ],
      [`embd=${this.model.n_embd}`, `heads=${this.model.n_head}`],
      { tokenIds: contextIds, tokenTexts: tokensText }
    );
    this.#pushSection(trace, "Embeddings", "Residual stream initialized", [embeddingNode]);

    for (let layer = 0; layer < this.model.n_layer; layer += 1) {
      const layerPrefix = `transformer.h.${layer}`;
      const layerNodes = [];

      const ln1 = layerNorm2D(
        x,
        contextIds.length,
        this.model.n_embd,
        this.#tensor(`${layerPrefix}.ln_1.weight`).data,
        this.#maybeTensor(`${layerPrefix}.ln_1.bias`)?.data
      );
      layerNodes.push(
        this.#makeNode(
          `layer.${layer}.ln1`,
          `Layer ${layer} · LN 1`,
          "Normalize the residual stream before attention",
          "rgba(121, 240, 217, 0.18)",
          [
            makeTensor("input", x, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
            makeTensor("output", ln1, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
          ],
          [`shape ${contextIds.length}×${this.model.n_embd}`]
        )
      );

      const cAttn = this.#tensor(`${layerPrefix}.attn.c_attn.weight`);
      const cAttnBias = this.#maybeTensor(`${layerPrefix}.attn.c_attn.bias`)?.data;
      const qkv = linear2D(ln1, contextIds.length, this.model.n_embd, cAttn.data, cAttn.shape[0], cAttnBias);
      const [q, k, v] = splitLastDim3(qkv, contextIds.length, this.model.n_embd);
      const qHeads = splitHeads(q, contextIds.length, this.model.n_head, this.model.head_size);
      const kHeads = splitHeads(k, contextIds.length, this.model.n_head, this.model.head_size);
      const vHeads = splitHeads(v, contextIds.length, this.model.n_head, this.model.head_size);
      const attention = causalAttention(
        qHeads,
        kHeads,
        vHeads,
        this.model.n_head,
        contextIds.length,
        this.model.head_size
      );
      const cProj = this.#tensor(`${layerPrefix}.attn.c_proj.weight`);
      const cProjBias = this.#maybeTensor(`${layerPrefix}.attn.c_proj.bias`)?.data;
      const attnOut = linear2D(
        attention.merged,
        contextIds.length,
        this.model.n_embd,
        cProj.data,
        cProj.shape[0],
        cProjBias
      );
      const xAfterAttn = addSameShape(x, attnOut);

      layerNodes.push(
        this.#makeNode(
          `layer.${layer}.attention`,
          `Layer ${layer} · Attention`,
          "Project QKV, apply the causal mask, mix value streams, and project back",
          "rgba(255, 184, 112, 0.2)",
          [
            makeTensor("qkv_projection", qkv, [contextIds.length, this.model.n_embd * 3], { semantic: "heatmap" }),
            makeTensor("queries", qHeads, [this.model.n_head, contextIds.length, this.model.head_size], { semantic: "heatmap" }),
            makeTensor("keys", kHeads, [this.model.n_head, contextIds.length, this.model.head_size], { semantic: "heatmap" }),
            makeTensor("values", vHeads, [this.model.n_head, contextIds.length, this.model.head_size], { semantic: "heatmap" }),
            makeTensor("scores_raw", attention.rawScores, [this.model.n_head, contextIds.length, contextIds.length], { semantic: "heatmap" }),
            makeTensor("scores_masked", attention.maskedScores, [this.model.n_head, contextIds.length, contextIds.length], { semantic: "heatmap" }),
            makeTensor("attention_probs", attention.probs, [this.model.n_head, contextIds.length, contextIds.length], { semantic: "heatmap" }),
            makeTensor("head_outputs", attention.headOutputs, [this.model.n_head, contextIds.length, this.model.head_size], { semantic: "heatmap" }),
            makeTensor("merged_heads", attention.merged, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
            makeTensor("projected_output", attnOut, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
          ],
          [`${this.model.n_head} heads`, `head size ${this.model.head_size}`],
          { tokenIds: contextIds, tokenTexts: tokensText }
        )
      );

      layerNodes.push(
        this.#makeNode(
          `layer.${layer}.residual1`,
          `Layer ${layer} · Residual Add`,
          "Attention output added back into the stream",
          "rgba(146, 167, 255, 0.18)",
          [
            makeTensor("residual_before", x, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
            makeTensor("attention_output", attnOut, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
            makeTensor("residual_after", xAfterAttn, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
          ],
          ["post-attention stream"]
        )
      );

      const ln2 = layerNorm2D(
        xAfterAttn,
        contextIds.length,
        this.model.n_embd,
        this.#tensor(`${layerPrefix}.ln_2.weight`).data,
        this.#maybeTensor(`${layerPrefix}.ln_2.bias`)?.data
      );
      layerNodes.push(
        this.#makeNode(
          `layer.${layer}.ln2`,
          `Layer ${layer} · LN 2`,
          "Normalize before the feed-forward network",
          "rgba(121, 240, 217, 0.18)",
          [
            makeTensor("input", xAfterAttn, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
            makeTensor("output", ln2, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
          ],
          [`mlp hidden ${this.model.mlp_hidden_size}`]
        )
      );

      const fc = this.#tensor(`${layerPrefix}.mlp.c_fc.weight`);
      const fcBias = this.#maybeTensor(`${layerPrefix}.mlp.c_fc.bias`)?.data;
      const fcOut = linear2D(ln2, contextIds.length, this.model.n_embd, fc.data, fc.shape[0], fcBias);
      const geluOut = applyGelu(fcOut);
      const proj = this.#tensor(`${layerPrefix}.mlp.c_proj.weight`);
      const projBias = this.#maybeTensor(`${layerPrefix}.mlp.c_proj.bias`)?.data;
      const mlpOut = linear2D(geluOut, contextIds.length, fc.shape[0], proj.data, proj.shape[0], projBias);
      const xAfterMlp = addSameShape(xAfterAttn, mlpOut);

      layerNodes.push(
        this.#makeNode(
          `layer.${layer}.mlp`,
          `Layer ${layer} · MLP`,
          "Expand, bend through GELU, then project back into the stream",
          "rgba(255, 126, 140, 0.18)",
          [
            makeTensor("fc_out", fcOut, [contextIds.length, fc.shape[0]], { semantic: "heatmap" }),
            makeTensor("gelu", geluOut, [contextIds.length, fc.shape[0]], { semantic: "heatmap" }),
            makeTensor("mlp_out", mlpOut, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
          ],
          [`expansion ${fc.shape[0] / this.model.n_embd}x`]
        )
      );

      layerNodes.push(
        this.#makeNode(
          `layer.${layer}.residual2`,
          `Layer ${layer} · Residual Add`,
          "MLP output returns to the main stream",
          "rgba(146, 167, 255, 0.18)",
          [
            makeTensor("residual_before", xAfterAttn, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
            makeTensor("mlp_output", mlpOut, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
            makeTensor("residual_after", xAfterMlp, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
          ],
          ["layer output"]
        )
      );

      x = xAfterMlp;
      this.#pushSection(trace, `Block ${layer}`, `Transformer layer ${layer}`, layerNodes);
    }

    const finalNorm = layerNorm2D(
      x,
      contextIds.length,
      this.model.n_embd,
      this.#tensor("transformer.ln_f.weight").data,
      this.#maybeTensor("transformer.ln_f.bias")?.data
    );
    const lmHead = this.#tensor("lm_head.weight");
    const logits = linear2D(finalNorm, contextIds.length, this.model.n_embd, lmHead.data, lmHead.shape[0], null);
    const lastLogits = rowFrom2D(logits, contextIds.length, this.model.vocab_size, contextIds.length - 1);
    const scaledLogits = scaleArray(lastLogits, 1 / temperature);
    const filteredLogits = applyTopK(scaledLogits, topK);
    const probs = softmax1D(filteredLogits);
    const nextTokenId = strategy === "sample" ? sampleFromDistribution(probs, seed) : argmax(probs);
    const nextTokenText = this.tokenizer.tokenLabel(nextTokenId);
    const topPredictions = topProbabilities(probs, this.tokenizer, 12);
    const outputIds = contextIds.concat(nextTokenId);
    const outputText = this.tokenizer.decode(outputIds);

    const finalNodes = [
      this.#makeNode(
        "output.ln_f",
        "Final LayerNorm",
        "Normalize the final residual stream before logits",
        "rgba(121, 240, 217, 0.18)",
        [
          makeTensor("input", x, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
          makeTensor("output", finalNorm, [contextIds.length, this.model.n_embd], { semantic: "heatmap" }),
        ],
        ["ready for unembedding"]
      ),
      this.#makeNode(
        "output.logits",
        "Logits",
        "Project hidden states into vocabulary space",
        "rgba(255, 184, 112, 0.2)",
        [
          makeTensor("logits_all_positions", logits, [contextIds.length, this.model.vocab_size], { semantic: "heatmap" }),
          makeTensor("last_logits", lastLogits, [this.model.vocab_size], { semantic: "vector" }),
          makeTensor("scaled_last_logits", scaledLogits, [this.model.vocab_size], { semantic: "vector" }),
        ],
        [`vocab ${this.model.vocab_size}`]
      ),
      this.#makeNode(
        "output.sampler",
        "Sampler",
        "Apply top-k and softmax, then select the next token",
        "rgba(255, 126, 140, 0.18)",
        [
          makeTensor("filtered_logits", filteredLogits, [this.model.vocab_size], { semantic: "vector" }),
          makeTensor("probabilities", probs, [this.model.vocab_size], { semantic: "vector" }),
        ],
        [`next = ${nextTokenText}`, strategy === "sample" ? `seed ${seed}` : "deterministic"],
        {
          topPredictions,
          tokenIds: outputIds,
          tokenTexts: outputIds.map((id) => this.tokenizer.tokenLabel(id)),
          nextTokenId,
          nextTokenText,
        }
      ),
    ];
    this.#pushSection(trace, "Output", "Logits and token choice", finalNodes);

    trace.nextTokenId = nextTokenId;
    trace.nextTokenText = nextTokenText;
    trace.outputIds = outputIds;
    trace.outputText = outputText;
    trace.elapsedMs = performance.now() - startTime;
    return trace;
  }

  #pushSection(trace, title, meta, nodes) {
    trace.sections.push({ id: title.toLowerCase().replace(/\s+/g, "-"), title, meta, nodes });
    trace.nodes.push(...nodes);
  }

  #makeNode(id, label, subtitle, accent, tensors, metrics = [], meta = {}) {
    return { id, label, subtitle, accent, tensors, metrics, meta };
  }

  #tensor(name) {
    const tensor = this.tensorMap.get(name);
    if (!tensor) {
      throw new Error(`Missing tensor: ${name}`);
    }
    return tensor;
  }

  #maybeTensor(name) {
    return this.tensorMap.get(name);
  }
}

export function makeTensor(name, data, shape, options = {}) {
  return {
    name,
    data,
    shape,
    semantic: options.semantic ?? inferSemantic(shape),
  };
}

export function tensorStats(values) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let finiteCount = 0;

  for (const value of values) {
    if (!Number.isFinite(value)) {
      continue;
    }
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    finiteCount += 1;
  }

  return {
    min: finiteCount === 0 ? 0 : min,
    max: finiteCount === 0 ? 0 : max,
    mean: finiteCount === 0 ? 0 : sum / finiteCount,
    finiteCount,
  };
}

export function sliceTensor(tensor, selectors) {
  if (tensor.shape.length <= 2) {
    return {
      data: tensor.data,
      shape: tensor.shape,
    };
  }

  const dims = tensor.shape;
  const fixedDims = dims.length - 2;
  const indices = selectors.slice(0, fixedDims);
  const strides = computeStrides(dims);
  let base = 0;
  for (let index = 0; index < fixedDims; index += 1) {
    base += indices[index] * strides[index];
  }

  const rows = dims[dims.length - 2];
  const cols = dims[dims.length - 1];
  const result = new Float32Array(rows * cols);
  const rowStride = strides[dims.length - 2];
  const colStride = strides[dims.length - 1];

  let cursor = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      result[cursor] = tensor.data[base + row * rowStride + col * colStride];
      cursor += 1;
    }
  }

  return {
    data: result,
    shape: [rows, cols],
  };
}

function inferSemantic(shape) {
  if (shape.length === 1) {
    return "vector";
  }
  return "heatmap";
}

function computeStrides(shape) {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let index = shape.length - 1; index >= 0; index -= 1) {
    strides[index] = stride;
    stride *= shape[index];
  }
  return strides;
}

function embeddingLookup(weight, embd, ids) {
  const out = new Float32Array(ids.length * embd);
  for (let row = 0; row < ids.length; row += 1) {
    const srcOffset = ids[row] * embd;
    const dstOffset = row * embd;
    out.set(weight.subarray(srcOffset, srcOffset + embd), dstOffset);
  }
  return out;
}

function addSameShape(a, b) {
  const out = new Float32Array(a.length);
  for (let index = 0; index < a.length; index += 1) {
    out[index] = a[index] + b[index];
  }
  return out;
}

function scaleArray(values, scalar) {
  const out = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    out[index] = values[index] * scalar;
  }
  return out;
}

function rowFrom2D(data, rows, cols, rowIndex) {
  const start = rowIndex * cols;
  return new Float32Array(data.slice(start, start + cols));
}

function layerNorm2D(x, rows, cols, weight, bias) {
  const out = new Float32Array(x.length);
  for (let row = 0; row < rows; row += 1) {
    const offset = row * cols;
    let mean = 0;
    for (let col = 0; col < cols; col += 1) {
      mean += x[offset + col];
    }
    mean /= cols;

    let variance = 0;
    for (let col = 0; col < cols; col += 1) {
      const centered = x[offset + col] - mean;
      variance += centered * centered;
    }
    variance /= cols;
    const invStd = 1 / Math.sqrt(variance + EPSILON);

    for (let col = 0; col < cols; col += 1) {
      const normalized = (x[offset + col] - mean) * invStd;
      const shifted = normalized * weight[col] + (bias ? bias[col] : 0);
      out[offset + col] = shifted;
    }
  }
  return out;
}

function linear2D(x, rows, inDim, weight, outDim, bias) {
  const out = new Float32Array(rows * outDim);
  for (let row = 0; row < rows; row += 1) {
    const xOffset = row * inDim;
    const outOffset = row * outDim;
    for (let outIndex = 0; outIndex < outDim; outIndex += 1) {
      let sum = bias ? bias[outIndex] : 0;
      const weightOffset = outIndex * inDim;
      for (let inIndex = 0; inIndex < inDim; inIndex += 1) {
        sum += x[xOffset + inIndex] * weight[weightOffset + inIndex];
      }
      out[outOffset + outIndex] = sum;
    }
  }
  return out;
}

function splitLastDim3(qkv, rows, width) {
  const q = new Float32Array(rows * width);
  const k = new Float32Array(rows * width);
  const v = new Float32Array(rows * width);
  for (let row = 0; row < rows; row += 1) {
    const src = row * width * 3;
    const dst = row * width;
    q.set(qkv.subarray(src, src + width), dst);
    k.set(qkv.subarray(src + width, src + width * 2), dst);
    v.set(qkv.subarray(src + width * 2, src + width * 3), dst);
  }
  return [q, k, v];
}

function splitHeads(data, seqLen, nHead, headSize) {
  const out = new Float32Array(nHead * seqLen * headSize);
  for (let token = 0; token < seqLen; token += 1) {
    for (let head = 0; head < nHead; head += 1) {
      const srcOffset = token * nHead * headSize + head * headSize;
      const dstOffset = head * seqLen * headSize + token * headSize;
      out.set(data.subarray(srcOffset, srcOffset + headSize), dstOffset);
    }
  }
  return out;
}

function causalAttention(q, k, v, nHead, seqLen, headSize) {
  const rawScores = new Float32Array(nHead * seqLen * seqLen);
  const maskedScores = new Float32Array(nHead * seqLen * seqLen);
  const probs = new Float32Array(nHead * seqLen * seqLen);
  const headOutputs = new Float32Array(nHead * seqLen * headSize);
  const scale = 1 / Math.sqrt(headSize);

  for (let head = 0; head < nHead; head += 1) {
    const headBase = head * seqLen * headSize;
    const scoreBase = head * seqLen * seqLen;
    for (let row = 0; row < seqLen; row += 1) {
      const qOffset = headBase + row * headSize;
      let maxScore = Number.NEGATIVE_INFINITY;
      for (let col = 0; col < seqLen; col += 1) {
        const kOffset = headBase + col * headSize;
        let dot = 0;
        for (let dim = 0; dim < headSize; dim += 1) {
          dot += q[qOffset + dim] * k[kOffset + dim];
        }
        const raw = dot * scale;
        const scoreIndex = scoreBase + row * seqLen + col;
        rawScores[scoreIndex] = raw;
        const masked = col <= row ? raw : Number.NEGATIVE_INFINITY;
        maskedScores[scoreIndex] = masked;
        if (masked > maxScore) {
          maxScore = masked;
        }
      }

      let denom = 0;
      for (let col = 0; col < seqLen; col += 1) {
        const scoreIndex = scoreBase + row * seqLen + col;
        const masked = maskedScores[scoreIndex];
        const value = Number.isFinite(masked) ? Math.exp(masked - maxScore) : 0;
        probs[scoreIndex] = value;
        denom += value;
      }
      denom = denom || 1;

      for (let col = 0; col < seqLen; col += 1) {
        const scoreIndex = scoreBase + row * seqLen + col;
        probs[scoreIndex] /= denom;
      }

      const outOffset = headBase + row * headSize;
      for (let dim = 0; dim < headSize; dim += 1) {
        let sum = 0;
        for (let col = 0; col < seqLen; col += 1) {
          const prob = probs[scoreBase + row * seqLen + col];
          const vOffset = headBase + col * headSize + dim;
          sum += prob * v[vOffset];
        }
        headOutputs[outOffset + dim] = sum;
      }
    }
  }

  const merged = new Float32Array(seqLen * nHead * headSize);
  for (let token = 0; token < seqLen; token += 1) {
    for (let head = 0; head < nHead; head += 1) {
      const srcOffset = head * seqLen * headSize + token * headSize;
      const dstOffset = token * nHead * headSize + head * headSize;
      merged.set(headOutputs.subarray(srcOffset, srcOffset + headSize), dstOffset);
    }
  }

  return { rawScores, maskedScores, probs, headOutputs, merged };
}

function applyGelu(values) {
  const out = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    out[index] = 0.5 * value * (1 + erf(value / Math.sqrt(2)));
  }
  return out;
}

function erf(value) {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x));
  return sign * y;
}

function applyTopK(logits, topK) {
  if (!topK || topK >= logits.length) {
    return new Float32Array(logits);
  }

  const sorted = Array.from(logits).sort((a, b) => b - a);
  const threshold = sorted[Math.max(0, topK - 1)];
  const out = new Float32Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    out[index] = logits[index] < threshold ? Number.NEGATIVE_INFINITY : logits[index];
  }
  return out;
}

function softmax1D(logits) {
  let max = Number.NEGATIVE_INFINITY;
  for (const value of logits) {
    if (value > max) {
      max = value;
    }
  }
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let index = 0; index < logits.length; index += 1) {
    const value = Number.isFinite(logits[index]) ? Math.exp(logits[index] - max) : 0;
    out[index] = value;
    sum += value;
  }
  sum = sum || 1;
  for (let index = 0; index < out.length; index += 1) {
    out[index] /= sum;
  }
  return out;
}

function argmax(values) {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] > bestValue) {
      bestValue = values[index];
      bestIndex = index;
    }
  }
  return bestIndex;
}

function sampleFromDistribution(probs, seed) {
  const rng = mulberry32(seed >>> 0);
  const target = rng();
  let cumulative = 0;
  for (let index = 0; index < probs.length; index += 1) {
    cumulative += probs[index];
    if (target <= cumulative) {
      return index;
    }
  }
  return probs.length - 1;
}

function mulberry32(seed) {
  let state = seed;
  return function next() {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function topProbabilities(probs, tokenizer, count) {
  return Array.from(probs)
    .map((prob, id) => ({ id, prob, token: tokenizer.tokenLabel(id) }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, count);
}
