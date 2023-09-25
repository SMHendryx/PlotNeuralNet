import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
offset="(0,0,0)"
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # args are name, out width, out channels
    to_Conv("conv1", 128, 32, offset=offset, to="(0,0,0)", height=64, depth=64, width=2 ),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
]

offset="(1,0,0)"
arch.extend([

    to_Conv("conv2", 64, 32, offset=offset, to="(pool1-east)", height=32, depth=32, width=2 ),
    to_connection( "pool1", "conv2"),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=16, depth=16, width=1),

    to_Conv("conv3", 32, 64, offset=offset, to="(pool2-east)", height=16, depth=16, width=4 ),
    to_connection( "pool2", "conv3"), 
    to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)", height=8, depth=8, width=1),

    to_Conv("conv4", 16, 128, offset=offset, to="(pool3-east)", height=8, depth=8, width=8),
    to_connection( "pool3", "conv4"), 
    to_Pool("pool4", offset="(0,0,0)", to="(conv4-east)", height=1, depth=1, width=1),



    to_Flatten("flatten", 120, offset, "(pool4-east)", caption="Flatten"  ),
    to_FullyConnected("linear1", 120, offset, "(flatten-east)", caption="Linear"  ),
    to_FullyConnected("linear2", 84 ,offset, "(linear1-east)", caption="Linear"  ),

    to_SoftMax("softmax", 4 ,offset, "(linear2-east)", caption="Softmax"  ),
    to_connection("pool4", "softmax"),    
    # to_Sum("sum1", offset="(1.5,0,0)", to="(softmax-east)", radius=2.5, opacity=0.6),
    # to_connection("softmax", "sum1"),
    to_end()
    ]
)

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()

