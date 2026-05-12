using System.IO;
using System.IO.Compression;

namespace KernelSharp;

/// <summary>
/// Shared helper used by every generated kernel launcher (.g.cs) to decode
/// the embedded fatbin byte array, which may be stored raw or gzip-compressed.
/// Centralising this avoids duplicating identical decompression logic in every
/// generated file.
/// </summary>
public static class FatbinHelper
{
    /// <summary>
    /// Decodes a fatbin that was embedded at build time.
    /// </summary>
    /// <param name="encoded">
    /// The embedded bytes — either the raw fatbin or its gzip-compressed form.
    /// </param>
    /// <param name="compression">
    /// The compression format used at build time: <c>"gzip"</c> or <c>"none"</c>.
    /// </param>
    /// <returns>The raw fatbin bytes ready to pass to <c>cuModuleLoadData</c>.</returns>
    public static byte[] Decode(byte[] encoded, string compression)
    {
        if (compression != "gzip")
        {
            return encoded;
        }

        using MemoryStream ms = new MemoryStream(encoded);
        using GZipStream gz = new GZipStream(ms, CompressionMode.Decompress);
        using MemoryStream output = new MemoryStream();
        gz.CopyTo(output);
        return output.ToArray();
    }
}
