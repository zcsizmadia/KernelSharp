using System.IO;
using System.IO.Compression;

namespace KernelSharp;

/// <summary>
/// Shared helper used by every generated kernel launcher (.g.cs) to decode
/// the embedded PTX byte array, which may be stored raw or compressed.
/// Centralising this avoids duplicating decompression logic in every generated file.
/// </summary>
public static class KernelBlobHelper
{
    /// <summary>
    /// Decodes a PTX blob that was embedded at build time.
    /// </summary>
    /// <param name="encoded">
    /// The embedded bytes — either raw PTX or its compressed form.
    /// </param>
    /// <param name="compression">
    /// The compression format used at build time:
    /// <c>"brotli"</c>, <c>"gzip"</c>, <c>"zlib"</c>, <c>"deflate"</c>, or <c>"none"</c>.
    /// </param>
    /// <returns>The raw PTX bytes ready to pass to <c>cuModuleLoadData</c>.</returns>
    public static byte[] Decode(byte[] encoded, string compression)
    {
        using MemoryStream input = new MemoryStream(encoded);
        using MemoryStream output = new MemoryStream();

        switch (compression)
        {
            case "brotli":
                using (var s = new BrotliStream(input, CompressionMode.Decompress))
                    s.CopyTo(output);
                break;
            case "gzip":
                using (var s = new GZipStream(input, CompressionMode.Decompress))
                    s.CopyTo(output);
                break;
            case "zlib":
                using (var s = new ZLibStream(input, CompressionMode.Decompress))
                    s.CopyTo(output);
                break;
            case "deflate":
                using (var s = new DeflateStream(input, CompressionMode.Decompress))
                    s.CopyTo(output);
                break;
            default: // "none" or any unrecognised value — return as-is
                return encoded;
        }

        return output.ToArray();
    }
}
