#[stage,(block structure, block number)]
# (block comp 1, block comp 2, ...)
# block comp (k1,k2,output_channels)

layout_18 = ((((3, 3, 64), (3, 3, 64)), 2),
             (((3, 3, 128), (3, 3, 128)), 2),
             (((3, 3, 256), (3, 3, 256)), 2),
             (((3, 3, 512), (3, 3, 512)), 2))
layout_34 = ((((3, 3, 64), (3, 3, 64)), 3),
             (((3, 3, 128), (3, 3, 128)), 4),
             (((3, 3, 256), (3, 3, 256)), 6),
             (((3, 3, 512), (3, 3, 512)), 3))
layout_50 = ((((1, 1, 64), (3, 3, 64), (1, 1, 256)), 3),
             (((1, 1, 128), (3, 3, 128), (1, 1, 512)), 4),
             (((1, 1, 256), (3, 3, 256), (1, 1, 1024)), 6),
             (((1, 1, 512), (3, 3, 512), (1, 1, 2048)), 3))
layout_101 = ((((1, 1, 64), (3, 3, 64), (1, 1, 256)), 3),
              (((1, 1, 128), (3, 3, 128), (1, 1, 512)), 4),
              (((1, 1, 256), (3, 3, 256), (1, 1, 1024)), 23),
              (((1, 1, 512), (3, 3, 512), (1, 1, 2048)), 3))
layout_152 = ((((1, 1, 64), (3, 3, 64), (1, 1, 256)), 3),
              (((1, 1, 128), (3, 3, 128), (1, 1, 512)), 8),
              (((1, 1, 256), (3, 3, 256), (1, 1, 1024)), 36),
              (((1, 1, 512), (3, 3, 512), (1, 1, 2048)), 3))

layouts = {18: layout_18, 34: layout_34, 50: layout_50, 101: layout_101, 152: layout_152}
