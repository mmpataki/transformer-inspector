const MAGIC_TEXT = "NGVIZ01\n";
const MAGIC_BYTES = new TextEncoder().encode(MAGIC_TEXT);

export async function loadBundle(file) {
  const buffer = await file.arrayBuffer();
  return parseBundle(buffer, file.name);
}

export function parseBundle(buffer, sourceName = "bundle") {
  const bytes = new Uint8Array(buffer, 0, MAGIC_BYTES.length);
  for (let index = 0; index < MAGIC_BYTES.length; index += 1) {
    if (bytes[index] !== MAGIC_BYTES[index]) {
      throw new Error("Unsupported bundle format. Re-export with tools/convert_checkpoint.py.");
    }
  }

  const view = new DataView(buffer);
  const manifestLength = view.getUint32(MAGIC_BYTES.length, true);
  const manifestStart = MAGIC_BYTES.length + 4;
  const manifestBytes = new Uint8Array(buffer, manifestStart, manifestLength);
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
  const payloadStart = manifestStart + manifestLength;

  const tensorMap = new Map();
  for (const record of manifest.model.tensors) {
    const byteOffset = payloadStart + record.offset;
    const data =
      byteOffset % 4 === 0
        ? new Float32Array(buffer, byteOffset, record.numel)
        : new Float32Array(buffer.slice(byteOffset, byteOffset + record.nbytes));
    tensorMap.set(record.name, { ...record, data });
  }

  return {
    sourceName,
    buffer,
    manifest,
    tensorMap,
  };
}

export function createTokenizer(tokenizerManifest) {
  if (!tokenizerManifest) {
    return {
      kind: "ids",
      promptHelp: "Enter comma-separated token IDs",
      encode(text) {
        return parseTokenIds(text);
      },
      decode(ids) {
        return ids.join(", ");
      },
      tokenLabel(id) {
        return String(id);
      },
    };
  }

  const itos = tokenizerManifest.itos;
  const stoi = new Map(itos.map((token, index) => [token, index]));
  const specialIds = tokenizerManifest.special_ids ?? {};

  if (tokenizerManifest.kind === "word") {
    const tokenRegex = new RegExp(tokenizerManifest.token_regex, "g");
    return {
      kind: "word",
      itos,
      stoi,
      specialIds,
      promptHelp: "Enter plain text",
      encode(text) {
        const matches = text.match(tokenRegex) ?? [];
        const unkId = specialIds["<unk>"];
        return matches.map((token) => {
          if (token === "<|endoftext|>" && specialIds["<eos>"] !== undefined) {
            return specialIds["<eos>"];
          }
          const id = stoi.get(token);
          if (id !== undefined) {
            return id;
          }
          if (unkId !== undefined) {
            return unkId;
          }
          throw new Error(`Token "${token}" not found and no <unk> token is available.`);
        });
      },
      decode(ids) {
        const tokens = ids.map((id) => itos[id] ?? `<${id}>`);
        const noSpaceBefore = new Set([".", ",", "!", "?", ";", ":", "%", ")", "]", "}"]);
        const noSpaceAfter = new Set(["(", "[", "{", "$"]);
        const parts = [];
        for (const token of tokens) {
          if (token === "<eos>") {
            parts.push("\n\n");
          } else if (token === "<unk>") {
            parts.push("<unk>");
          } else if (
            parts.length === 0 ||
            parts[parts.length - 1].endsWith("\n\n") ||
            noSpaceBefore.has(token)
          ) {
            parts.push(token);
          } else if (noSpaceAfter.has(parts[parts.length - 1])) {
            parts.push(token);
          } else {
            parts.push(` ${token}`);
          }
        }
        return parts.join("");
      },
      tokenLabel(id) {
        return itos[id] ?? `<${id}>`;
      },
    };
  }

  return {
    kind: "char",
    itos,
    stoi,
    specialIds,
    promptHelp: "Enter plain text",
    encode(text) {
      return Array.from(text).map((character) => {
        const id = stoi.get(character);
        if (id === undefined) {
          throw new Error(`Character "${character}" is outside this tokenizer vocabulary.`);
        }
        return id;
      });
    },
    decode(ids) {
      return ids.map((id) => itos[id] ?? "").join("");
    },
    tokenLabel(id) {
      const token = itos[id];
      if (token === "\n") {
        return "\\n";
      }
      if (token === "\t") {
        return "\\t";
      }
      return token ?? `<${id}>`;
    },
  };
}

function parseTokenIds(text) {
  return text
    .split(/[\s,]+/)
    .map((value) => value.trim())
    .filter(Boolean)
    .map((value) => {
      const id = Number.parseInt(value, 10);
      if (!Number.isFinite(id)) {
        throw new Error(`Invalid token id: ${value}`);
      }
      return id;
    });
}
