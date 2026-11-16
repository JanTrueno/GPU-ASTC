# Third-party notices

## niepp/astc_encoder (MIT)

`shaders/astc_encode.comp` is ported from the HLSL compute shaders in https://github.com/niepp/astc_encoder
(Copyright (c) 2021 niedap). The full licence text is in
`third_party/niepp_astc_encoder_LICENSE`. The original lookup tables were replaced
by arithmetic (trit packing derived from the ASTC spec's decode rules, checked against
the original table for all 243 combinations).

## stb_image (public domain / MIT)

`third_party/stb_image.h` is Sean Barrett's single-file image loader
(version 2.22 as vendored from the project above). It should be updated to a
current release.
