��
�<�;
:
Add
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint�
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
�
CTCBeamSearchDecoder

inputs
sequence_length
decoded_indices	*	top_paths
decoded_values	*	top_paths
decoded_shape	*	top_paths
log_probability"

beam_widthint(0"
	top_pathsint(0"
merge_repeatedbool(
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

9
	DecodeBmp
contents	
image"
channelsint 
$
	DecodeGif
contents	
image
�

DecodeJpeg
contents	
image"
channelsint "
ratioint"
fancy_upscalingbool("!
try_recover_truncatedbool( "#
acceptable_fractionfloat%  �?"

dct_methodstring 
Y
	DecodePng
contents
image"dtype"
channelsint "
dtypetype0:
2
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	
�
)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
�
FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%��8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
\
ListDiff
x"T
y"T
out"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
$

LogicalAnd
x

y

z
�
!
LoopCond	
input


output

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
�
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
q
ResizeBilinear
images"T
size
resized_images"
Ttype:
2
	"
align_cornersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
s
Substr	
input
pos"T
len"T

output"
Ttype:
2	"+
unitstringBYTE:
BYTE	UTF8_CHAR
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:�
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype�
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype�
9
TensorArraySizeV3

handle
flow_in
size�
�
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring �
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype�
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �"serve*1.13.12
b'unknown'��	
P
PlaceholderPlaceholder*
dtype0*
_output_shapes
:*
shape:
=
xIdentityPlaceholder*
T0*
_output_shapes
:
Y
decode_image/Substr/posConst*
value	B : *
dtype0*
_output_shapes
: 
Y
decode_image/Substr/lenConst*
dtype0*
_output_shapes
: *
value	B :
�
decode_image/SubstrSubstrxdecode_image/Substr/posdecode_image/Substr/len*
T0*
_output_shapes
:*
unitBYTE
a
decode_image/is_jpeg/Substr/posConst*
value	B : *
dtype0*
_output_shapes
: 
a
decode_image/is_jpeg/Substr/lenConst*
value	B :*
dtype0*
_output_shapes
: 
�
decode_image/is_jpeg/SubstrSubstrxdecode_image/is_jpeg/Substr/posdecode_image/is_jpeg/Substr/len*
_output_shapes
:*
unitBYTE*
T0
`
decode_image/is_jpeg/Equal/yConst*
valueB	 B���*
dtype0*
_output_shapes
: 
�
decode_image/is_jpeg/EqualEqualdecode_image/is_jpeg/Substrdecode_image/is_jpeg/Equal/y*
_output_shapes
:*
T0
�
decode_image/cond_jpeg/SwitchSwitchdecode_image/is_jpeg/Equaldecode_image/is_jpeg/Equal*
T0
*
_output_shapes

::
o
decode_image/cond_jpeg/switch_tIdentitydecode_image/cond_jpeg/Switch:1*
_output_shapes
:*
T0

m
decode_image/cond_jpeg/switch_fIdentitydecode_image/cond_jpeg/Switch*
T0
*
_output_shapes
:
i
decode_image/cond_jpeg/pred_idIdentitydecode_image/is_jpeg/Equal*
_output_shapes
:*
T0

�
,decode_image/cond_jpeg/check_jpeg_channels/xConst ^decode_image/cond_jpeg/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
,decode_image/cond_jpeg/check_jpeg_channels/yConst ^decode_image/cond_jpeg/switch_t*
dtype0*
_output_shapes
: *
value	B :
�
*decode_image/cond_jpeg/check_jpeg_channelsNotEqual,decode_image/cond_jpeg/check_jpeg_channels/x,decode_image/cond_jpeg/check_jpeg_channels/y*
T0*
_output_shapes
: 
�
#decode_image/cond_jpeg/Assert/ConstConst ^decode_image/cond_jpeg/switch_t*N
valueEBC B=Channels must be in (None, 0, 1, 3) when decoding JPEG images*
dtype0*
_output_shapes
: 
�
+decode_image/cond_jpeg/Assert/Assert/data_0Const ^decode_image/cond_jpeg/switch_t*N
valueEBC B=Channels must be in (None, 0, 1, 3) when decoding JPEG images*
dtype0*
_output_shapes
: 
�
$decode_image/cond_jpeg/Assert/AssertAssert*decode_image/cond_jpeg/check_jpeg_channels+decode_image/cond_jpeg/Assert/Assert/data_0*

T
2*
	summarize
�
!decode_image/cond_jpeg/DecodeJpeg
DecodeJpeg*decode_image/cond_jpeg/DecodeJpeg/Switch:1%^decode_image/cond_jpeg/Assert/Assert*

dct_method *
channels*
acceptable_fraction%  �?*
fancy_upscaling(*
try_recover_truncated( *4
_output_shapes"
 :������������������*
ratio
�
(decode_image/cond_jpeg/DecodeJpeg/SwitchSwitchxdecode_image/cond_jpeg/pred_id*
T0*
_class

loc:@x*
_output_shapes

::
�
decode_image/cond_jpeg/IdentityIdentity!decode_image/cond_jpeg/DecodeJpeg*
T0*4
_output_shapes"
 :������������������
�
(decode_image/cond_jpeg/is_png/Substr/posConst ^decode_image/cond_jpeg/switch_f*
value	B : *
dtype0*
_output_shapes
: 
�
(decode_image/cond_jpeg/is_png/Substr/lenConst ^decode_image/cond_jpeg/switch_f*
value	B :*
dtype0*
_output_shapes
: 
�
$decode_image/cond_jpeg/is_png/SubstrSubstr+decode_image/cond_jpeg/is_png/Substr/Switch(decode_image/cond_jpeg/is_png/Substr/pos(decode_image/cond_jpeg/is_png/Substr/len*
_output_shapes
:*
unitBYTE*
T0
�
+decode_image/cond_jpeg/is_png/Substr/SwitchSwitchxdecode_image/cond_jpeg/pred_id*
T0*
_class

loc:@x*
_output_shapes

::
�
%decode_image/cond_jpeg/is_png/Equal/yConst ^decode_image/cond_jpeg/switch_f*
valueB	 B�PN*
dtype0*
_output_shapes
: 
�
#decode_image/cond_jpeg/is_png/EqualEqual$decode_image/cond_jpeg/is_png/Substr%decode_image/cond_jpeg/is_png/Equal/y*
_output_shapes
:*
T0
�
&decode_image/cond_jpeg/cond_png/SwitchSwitch#decode_image/cond_jpeg/is_png/Equal#decode_image/cond_jpeg/is_png/Equal*
T0
*
_output_shapes

::
�
(decode_image/cond_jpeg/cond_png/switch_tIdentity(decode_image/cond_jpeg/cond_png/Switch:1*
T0
*
_output_shapes
:

(decode_image/cond_jpeg/cond_png/switch_fIdentity&decode_image/cond_jpeg/cond_png/Switch*
T0
*
_output_shapes
:
{
'decode_image/cond_jpeg/cond_png/pred_idIdentity#decode_image/cond_jpeg/is_png/Equal*
T0
*
_output_shapes
:
�
)decode_image/cond_jpeg/cond_png/DecodePng	DecodePng2decode_image/cond_jpeg/cond_png/DecodePng/Switch:1*
channels*
dtype0*4
_output_shapes"
 :������������������
�
0decode_image/cond_jpeg/cond_png/DecodePng/SwitchSwitch+decode_image/cond_jpeg/is_png/Substr/Switch'decode_image/cond_jpeg/cond_png/pred_id*
T0*
_class

loc:@x*
_output_shapes

::
�
(decode_image/cond_jpeg/cond_png/IdentityIdentity)decode_image/cond_jpeg/cond_png/DecodePng*
T0*4
_output_shapes"
 :������������������
�
(decode_image/cond_jpeg/cond_png/is_gif/yConst)^decode_image/cond_jpeg/cond_png/switch_f*
valueB	 BGIF*
dtype0*
_output_shapes
: 
�
&decode_image/cond_jpeg/cond_png/is_gifEqual/decode_image/cond_jpeg/cond_png/is_gif/Switch_1(decode_image/cond_jpeg/cond_png/is_gif/y*
_output_shapes
:*
T0
�
-decode_image/cond_jpeg/cond_png/is_gif/SwitchSwitchdecode_image/Substrdecode_image/cond_jpeg/pred_id*
T0*&
_class
loc:@decode_image/Substr*
_output_shapes

::
�
/decode_image/cond_jpeg/cond_png/is_gif/Switch_1Switch-decode_image/cond_jpeg/cond_png/is_gif/Switch'decode_image/cond_jpeg/cond_png/pred_id*
T0*&
_class
loc:@decode_image/Substr*
_output_shapes

::
�
/decode_image/cond_jpeg/cond_png/cond_gif/SwitchSwitch&decode_image/cond_jpeg/cond_png/is_gif&decode_image/cond_jpeg/cond_png/is_gif*
_output_shapes

::*
T0

�
1decode_image/cond_jpeg/cond_png/cond_gif/switch_tIdentity1decode_image/cond_jpeg/cond_png/cond_gif/Switch:1*
T0
*
_output_shapes
:
�
1decode_image/cond_jpeg/cond_png/cond_gif/switch_fIdentity/decode_image/cond_jpeg/cond_png/cond_gif/Switch*
_output_shapes
:*
T0

�
0decode_image/cond_jpeg/cond_png/cond_gif/pred_idIdentity&decode_image/cond_jpeg/cond_png/is_gif*
T0
*
_output_shapes
:
�
=decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/xConst2^decode_image/cond_jpeg/cond_png/cond_gif/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
=decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/yConst2^decode_image/cond_jpeg/cond_png/cond_gif/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
;decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channelsNotEqual=decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/x=decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/y*
T0*
_output_shapes
: 
�
?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/xConst2^decode_image/cond_jpeg/cond_png/cond_gif/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/yConst2^decode_image/cond_jpeg/cond_png/cond_gif/switch_t*
value	B :*
dtype0*
_output_shapes
: 
�
=decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1NotEqual?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/x?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/y*
T0*
_output_shapes
: 
�
3decode_image/cond_jpeg/cond_png/cond_gif/LogicalAnd
LogicalAnd;decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels=decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1*
_output_shapes
: 
�
5decode_image/cond_jpeg/cond_png/cond_gif/Assert/ConstConst2^decode_image/cond_jpeg/cond_png/cond_gif/switch_t*
dtype0*
_output_shapes
: *J
valueAB? B9Channels must be in (None, 0, 3) when decoding GIF images
�
=decode_image/cond_jpeg/cond_png/cond_gif/Assert/Assert/data_0Const2^decode_image/cond_jpeg/cond_png/cond_gif/switch_t*J
valueAB? B9Channels must be in (None, 0, 3) when decoding GIF images*
dtype0*
_output_shapes
: 
�
6decode_image/cond_jpeg/cond_png/cond_gif/Assert/AssertAssert3decode_image/cond_jpeg/cond_png/cond_gif/LogicalAnd=decode_image/cond_jpeg/cond_png/cond_gif/Assert/Assert/data_0*

T
2*
	summarize
�
2decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif	DecodeGif=decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch_1:17^decode_image/cond_jpeg/cond_png/cond_gif/Assert/Assert*A
_output_shapes/
-:+���������������������������
�
9decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/SwitchSwitch+decode_image/cond_jpeg/is_png/Substr/Switch'decode_image/cond_jpeg/cond_png/pred_id*
T0*
_class

loc:@x*
_output_shapes

::
�
;decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch_1Switch9decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch0decode_image/cond_jpeg/cond_png/cond_gif/pred_id*
_output_shapes

::*
T0*
_class

loc:@x
�
1decode_image/cond_jpeg/cond_png/cond_gif/IdentityIdentity2decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif*
T0*A
_output_shapes/
-:+���������������������������
�
3decode_image/cond_jpeg/cond_png/cond_gif/Substr/posConst2^decode_image/cond_jpeg/cond_png/cond_gif/switch_f*
dtype0*
_output_shapes
: *
value	B : 
�
3decode_image/cond_jpeg/cond_png/cond_gif/Substr/lenConst2^decode_image/cond_jpeg/cond_png/cond_gif/switch_f*
value	B :*
dtype0*
_output_shapes
: 
�
/decode_image/cond_jpeg/cond_png/cond_gif/SubstrSubstr6decode_image/cond_jpeg/cond_png/cond_gif/Substr/Switch3decode_image/cond_jpeg/cond_png/cond_gif/Substr/pos3decode_image/cond_jpeg/cond_png/cond_gif/Substr/len*
T0*
_output_shapes
:*
unitBYTE
�
6decode_image/cond_jpeg/cond_png/cond_gif/Substr/SwitchSwitch9decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch0decode_image/cond_jpeg/cond_png/cond_gif/pred_id*
T0*
_class

loc:@x*
_output_shapes

::
�
1decode_image/cond_jpeg/cond_png/cond_gif/is_bmp/yConst2^decode_image/cond_jpeg/cond_png/cond_gif/switch_f*
value
B BBM*
dtype0*
_output_shapes
: 
�
/decode_image/cond_jpeg/cond_png/cond_gif/is_bmpEqual/decode_image/cond_jpeg/cond_png/cond_gif/Substr1decode_image/cond_jpeg/cond_png/cond_gif/is_bmp/y*
T0*
_output_shapes
:
�
7decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/ConstConst2^decode_image/cond_jpeg/cond_png/cond_gif/switch_f*A
value8B6 B0Unable to decode bytes as JPEG, PNG, GIF, or BMP*
dtype0*
_output_shapes
: 
�
?decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Assert/data_0Const2^decode_image/cond_jpeg/cond_png/cond_gif/switch_f*A
value8B6 B0Unable to decode bytes as JPEG, PNG, GIF, or BMP*
dtype0*
_output_shapes
: 
�
8decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/AssertAssert/decode_image/cond_jpeg/cond_png/cond_gif/is_bmp?decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Assert/data_0*

T
2*
	summarize
�
9decode_image/cond_jpeg/cond_png/cond_gif/check_channels/xConst2^decode_image/cond_jpeg/cond_png/cond_gif/switch_f*
value	B :*
dtype0*
_output_shapes
: 
�
9decode_image/cond_jpeg/cond_png/cond_gif/check_channels/yConst2^decode_image/cond_jpeg/cond_png/cond_gif/switch_f*
value	B :*
dtype0*
_output_shapes
: 
�
7decode_image/cond_jpeg/cond_png/cond_gif/check_channelsNotEqual9decode_image/cond_jpeg/cond_png/cond_gif/check_channels/x9decode_image/cond_jpeg/cond_png/cond_gif/check_channels/y*
T0*
_output_shapes
: 
�
7decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/ConstConst2^decode_image/cond_jpeg/cond_png/cond_gif/switch_f*J
valueAB? B9Channels must be in (None, 0, 3) when decoding BMP images*
dtype0*
_output_shapes
: 
�
?decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Assert/data_0Const2^decode_image/cond_jpeg/cond_png/cond_gif/switch_f*J
valueAB? B9Channels must be in (None, 0, 3) when decoding BMP images*
dtype0*
_output_shapes
: 
�
8decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/AssertAssert7decode_image/cond_jpeg/cond_png/cond_gif/check_channels?decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Assert/data_0*

T
2*
	summarize
�
2decode_image/cond_jpeg/cond_png/cond_gif/DecodeBmp	DecodeBmp6decode_image/cond_jpeg/cond_png/cond_gif/Substr/Switch9^decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Assert9^decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Assert*
channels *=
_output_shapes+
):'���������������������������
�
3decode_image/cond_jpeg/cond_png/cond_gif/Identity_1Identity2decode_image/cond_jpeg/cond_png/cond_gif/DecodeBmp*
T0*=
_output_shapes+
):'���������������������������
�
.decode_image/cond_jpeg/cond_png/cond_gif/MergeMerge3decode_image/cond_jpeg/cond_png/cond_gif/Identity_11decode_image/cond_jpeg/cond_png/cond_gif/Identity*
N*
_output_shapes
:: *
T0
�
%decode_image/cond_jpeg/cond_png/MergeMerge.decode_image/cond_jpeg/cond_png/cond_gif/Merge(decode_image/cond_jpeg/cond_png/Identity*
N*
_output_shapes
:: *
T0
�
decode_image/cond_jpeg/MergeMerge%decode_image/cond_jpeg/cond_png/Mergedecode_image/cond_jpeg/Identity*
T0*
N*
_output_shapes
:: 
�
#rgb_to_grayscale/convert_image/CastCastdecode_image/cond_jpeg/Merge*

SrcT0*
Truncate( *
_output_shapes
:*

DstT0
e
 rgb_to_grayscale/convert_image/yConst*
valueB
 *���;*
dtype0*
_output_shapes
: 
�
rgb_to_grayscale/convert_imageMul#rgb_to_grayscale/convert_image/Cast rgb_to_grayscale/convert_image/y*
T0*
_output_shapes
:
q
rgb_to_grayscale/Tensordot/bConst*!
valueB"l	�>�E?�x�=*
dtype0*
_output_shapes
:
�
 rgb_to_grayscale/Tensordot/ShapeShapergb_to_grayscale/convert_image*
T0*
out_type0*#
_output_shapes
:���������
h
rgb_to_grayscale/Tensordot/RankRankrgb_to_grayscale/convert_image*
_output_shapes
: *
T0
r
rgb_to_grayscale/Tensordot/axesConst*
valueB:
���������*
dtype0*
_output_shapes
:
k
)rgb_to_grayscale/Tensordot/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
'rgb_to_grayscale/Tensordot/GreaterEqualGreaterEqualrgb_to_grayscale/Tensordot/axes)rgb_to_grayscale/Tensordot/GreaterEqual/y*
T0*
_output_shapes
:
�
rgb_to_grayscale/Tensordot/addAddrgb_to_grayscale/Tensordot/axesrgb_to_grayscale/Tensordot/Rank*
T0*
_output_shapes
:
�
!rgb_to_grayscale/Tensordot/SelectSelect'rgb_to_grayscale/Tensordot/GreaterEqualrgb_to_grayscale/Tensordot/axesrgb_to_grayscale/Tensordot/add*
T0*
_output_shapes
:
h
&rgb_to_grayscale/Tensordot/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
h
&rgb_to_grayscale/Tensordot/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
 rgb_to_grayscale/Tensordot/rangeRange&rgb_to_grayscale/Tensordot/range/startrgb_to_grayscale/Tensordot/Rank&rgb_to_grayscale/Tensordot/range/delta*

Tidx0*#
_output_shapes
:���������
�
#rgb_to_grayscale/Tensordot/ListDiffListDiff rgb_to_grayscale/Tensordot/range!rgb_to_grayscale/Tensordot/Select*2
_output_shapes 
:���������:���������*
out_idx0*
T0
j
(rgb_to_grayscale/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#rgb_to_grayscale/Tensordot/GatherV2GatherV2 rgb_to_grayscale/Tensordot/Shape#rgb_to_grayscale/Tensordot/ListDiff(rgb_to_grayscale/Tensordot/GatherV2/axis*
Tindices0*
Tparams0*#
_output_shapes
:���������*
Taxis0
l
*rgb_to_grayscale/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
%rgb_to_grayscale/Tensordot/GatherV2_1GatherV2 rgb_to_grayscale/Tensordot/Shape!rgb_to_grayscale/Tensordot/Select*rgb_to_grayscale/Tensordot/GatherV2_1/axis*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0
j
 rgb_to_grayscale/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
rgb_to_grayscale/Tensordot/ProdProd#rgb_to_grayscale/Tensordot/GatherV2 rgb_to_grayscale/Tensordot/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
l
"rgb_to_grayscale/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
!rgb_to_grayscale/Tensordot/Prod_1Prod%rgb_to_grayscale/Tensordot/GatherV2_1"rgb_to_grayscale/Tensordot/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
h
&rgb_to_grayscale/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
!rgb_to_grayscale/Tensordot/concatConcatV2#rgb_to_grayscale/Tensordot/ListDiff!rgb_to_grayscale/Tensordot/Select&rgb_to_grayscale/Tensordot/concat/axis*
T0*
N*#
_output_shapes
:���������*

Tidx0
�
 rgb_to_grayscale/Tensordot/stackPackrgb_to_grayscale/Tensordot/Prod!rgb_to_grayscale/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:
�
$rgb_to_grayscale/Tensordot/transpose	Transposergb_to_grayscale/convert_image!rgb_to_grayscale/Tensordot/concat*
_output_shapes
:*
Tperm0*
T0
�
"rgb_to_grayscale/Tensordot/ReshapeReshape$rgb_to_grayscale/Tensordot/transpose rgb_to_grayscale/Tensordot/stack*0
_output_shapes
:������������������*
T0*
Tshape0
u
+rgb_to_grayscale/Tensordot/transpose_1/permConst*
valueB: *
dtype0*
_output_shapes
:
�
&rgb_to_grayscale/Tensordot/transpose_1	Transposergb_to_grayscale/Tensordot/b+rgb_to_grayscale/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes
:
{
*rgb_to_grayscale/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
$rgb_to_grayscale/Tensordot/Reshape_1Reshape&rgb_to_grayscale/Tensordot/transpose_1*rgb_to_grayscale/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:
�
!rgb_to_grayscale/Tensordot/MatMulMatMul"rgb_to_grayscale/Tensordot/Reshape$rgb_to_grayscale/Tensordot/Reshape_1*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
e
"rgb_to_grayscale/Tensordot/Const_2Const*
dtype0*
_output_shapes
: *
valueB 
j
(rgb_to_grayscale/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#rgb_to_grayscale/Tensordot/concat_1ConcatV2#rgb_to_grayscale/Tensordot/GatherV2"rgb_to_grayscale/Tensordot/Const_2(rgb_to_grayscale/Tensordot/concat_1/axis*
N*#
_output_shapes
:���������*

Tidx0*
T0
�
rgb_to_grayscale/TensordotReshape!rgb_to_grayscale/Tensordot/MatMul#rgb_to_grayscale/Tensordot/concat_1*
T0*
Tshape0*
_output_shapes
:
j
rgb_to_grayscale/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
rgb_to_grayscale/ExpandDims
ExpandDimsrgb_to_grayscale/Tensordotrgb_to_grayscale/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
[
rgb_to_grayscale/Mul/yConst*
valueB
 * �C*
dtype0*
_output_shapes
: 
s
rgb_to_grayscale/MulMulrgb_to_grayscale/ExpandDimsrgb_to_grayscale/Mul/y*
T0*
_output_shapes
:
p
rgb_to_grayscaleCastrgb_to_grayscale/Mul*

SrcT0*
Truncate( *
_output_shapes
:*

DstT0
n
convert_image/CastCastrgb_to_grayscale*

SrcT0*
Truncate( *
_output_shapes
:*

DstT0
T
convert_image/yConst*
dtype0*
_output_shapes
: *
valueB
 *���;
\
convert_imageMulconvert_image/Castconvert_image/y*
_output_shapes
:*
T0
f
$resize_image_with_pad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
 resize_image_with_pad/ExpandDims
ExpandDimsconvert_image$resize_image_with_pad/ExpandDims/dim*

Tdim0*
T0*J
_output_shapes8
6:4������������������������������������
{
resize_image_with_pad/ShapeShape resize_image_with_pad/ExpandDims*
T0*
out_type0*
_output_shapes
:
m
+resize_image_with_pad/assert_positive/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
�
6resize_image_with_pad/assert_positive/assert_less/LessLess+resize_image_with_pad/assert_positive/Constresize_image_with_pad/Shape*
T0*
_output_shapes
:
�
7resize_image_with_pad/assert_positive/assert_less/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
5resize_image_with_pad/assert_positive/assert_less/AllAll6resize_image_with_pad/assert_positive/assert_less/Less7resize_image_with_pad/assert_positive/assert_less/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
>resize_image_with_pad/assert_positive/assert_less/Assert/ConstConst*7
value.B, B&all dims of 'image.shape' must be > 0.*
dtype0*
_output_shapes
: 
�
Fresize_image_with_pad/assert_positive/assert_less/Assert/Assert/data_0Const*7
value.B, B&all dims of 'image.shape' must be > 0.*
dtype0*
_output_shapes
: 
�
?resize_image_with_pad/assert_positive/assert_less/Assert/AssertAssert5resize_image_with_pad/assert_positive/assert_less/AllFresize_image_with_pad/assert_positive/assert_less/Assert/Assert/data_0*

T
2*
	summarize
�
(resize_image_with_pad/control_dependencyIdentity resize_image_with_pad/ExpandDims@^resize_image_with_pad/assert_positive/assert_less/Assert/Assert*J
_output_shapes8
6:4������������������������������������*
T0*3
_class)
'%loc:@resize_image_with_pad/ExpandDims
�
resize_image_with_pad/Shape_1Shape(resize_image_with_pad/control_dependency*
_output_shapes
:*
T0*
out_type0
�
resize_image_with_pad/unstackUnpackresize_image_with_pad/Shape_1*	
num*
T0*

axis *
_output_shapes

: : : : 
�
resize_image_with_pad/CastCastresize_image_with_pad/unstack:1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
resize_image_with_pad/Cast_1Castresize_image_with_pad/unstack:2*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
`
resize_image_with_pad/Cast_2/xConst*
value	B :.*
dtype0*
_output_shapes
: 
�
resize_image_with_pad/Cast_2Castresize_image_with_pad/Cast_2/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
a
resize_image_with_pad/Cast_3/xConst*
value
B :�*
dtype0*
_output_shapes
: 
�
resize_image_with_pad/Cast_3Castresize_image_with_pad/Cast_3/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
resize_image_with_pad/truedivRealDivresize_image_with_pad/Cast_1resize_image_with_pad/Cast_3*
T0*
_output_shapes
: 
�
resize_image_with_pad/truediv_1RealDivresize_image_with_pad/Castresize_image_with_pad/Cast_2*
T0*
_output_shapes
: 
�
resize_image_with_pad/MaximumMaximumresize_image_with_pad/truedivresize_image_with_pad/truediv_1*
T0*
_output_shapes
: 
�
resize_image_with_pad/truediv_2RealDivresize_image_with_pad/Castresize_image_with_pad/Maximum*
T0*
_output_shapes
: 
�
resize_image_with_pad/truediv_3RealDivresize_image_with_pad/Cast_1resize_image_with_pad/Maximum*
T0*
_output_shapes
: 
f
resize_image_with_pad/FloorFloorresize_image_with_pad/truediv_2*
T0*
_output_shapes
: 
�
resize_image_with_pad/Cast_4Castresize_image_with_pad/Floor*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
h
resize_image_with_pad/Floor_1Floorresize_image_with_pad/truediv_3*
T0*
_output_shapes
: 
�
resize_image_with_pad/Cast_5Castresize_image_with_pad/Floor_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
resize_image_with_pad/subSubresize_image_with_pad/Cast_2resize_image_with_pad/truediv_2*
T0*
_output_shapes
: 
j
!resize_image_with_pad/truediv_4/yConst*
valueB 2       @*
dtype0*
_output_shapes
: 
�
resize_image_with_pad/truediv_4RealDivresize_image_with_pad/sub!resize_image_with_pad/truediv_4/y*
T0*
_output_shapes
: 
�
resize_image_with_pad/sub_1Subresize_image_with_pad/Cast_3resize_image_with_pad/truediv_3*
T0*
_output_shapes
: 
j
!resize_image_with_pad/truediv_5/yConst*
dtype0*
_output_shapes
: *
valueB 2       @
�
resize_image_with_pad/truediv_5RealDivresize_image_with_pad/sub_1!resize_image_with_pad/truediv_5/y*
T0*
_output_shapes
: 
h
resize_image_with_pad/Floor_2Floorresize_image_with_pad/truediv_4*
T0*
_output_shapes
: 
h
resize_image_with_pad/Floor_3Floorresize_image_with_pad/truediv_5*
T0*
_output_shapes
: 
�
resize_image_with_pad/Cast_6Castresize_image_with_pad/Floor_2*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
c
!resize_image_with_pad/Maximum_1/xConst*
value	B : *
dtype0*
_output_shapes
: 
�
resize_image_with_pad/Maximum_1Maximum!resize_image_with_pad/Maximum_1/xresize_image_with_pad/Cast_6*
T0*
_output_shapes
: 
�
resize_image_with_pad/Cast_7Castresize_image_with_pad/Floor_3*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
c
!resize_image_with_pad/Maximum_2/xConst*
value	B : *
dtype0*
_output_shapes
: 
�
resize_image_with_pad/Maximum_2Maximum!resize_image_with_pad/Maximum_2/xresize_image_with_pad/Cast_7*
T0*
_output_shapes
: 
�
!resize_image_with_pad/resize/sizePackresize_image_with_pad/Cast_4resize_image_with_pad/Cast_5*
T0*

axis *
N*
_output_shapes
:
�
+resize_image_with_pad/resize/ResizeBilinearResizeBilinear(resize_image_with_pad/control_dependency!resize_image_with_pad/resize/size*
T0*J
_output_shapes8
6:4������������������������������������*
align_corners( 
�
/resize_image_with_pad/pad_to_bounding_box/ShapeShape+resize_image_with_pad/resize/ResizeBilinear*
T0*
out_type0*
_output_shapes
:
�
?resize_image_with_pad/pad_to_bounding_box/assert_positive/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
�
Jresize_image_with_pad/pad_to_bounding_box/assert_positive/assert_less/LessLess?resize_image_with_pad/pad_to_bounding_box/assert_positive/Const/resize_image_with_pad/pad_to_bounding_box/Shape*
T0*
_output_shapes
:
�
Kresize_image_with_pad/pad_to_bounding_box/assert_positive/assert_less/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
Iresize_image_with_pad/pad_to_bounding_box/assert_positive/assert_less/AllAllJresize_image_with_pad/pad_to_bounding_box/assert_positive/assert_less/LessKresize_image_with_pad/pad_to_bounding_box/assert_positive/assert_less/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
�
Rresize_image_with_pad/pad_to_bounding_box/assert_positive/assert_less/Assert/ConstConst*7
value.B, B&all dims of 'image.shape' must be > 0.*
dtype0*
_output_shapes
: 
�
Zresize_image_with_pad/pad_to_bounding_box/assert_positive/assert_less/Assert/Assert/data_0Const*7
value.B, B&all dims of 'image.shape' must be > 0.*
dtype0*
_output_shapes
: 
�
Sresize_image_with_pad/pad_to_bounding_box/assert_positive/assert_less/Assert/AssertAssertIresize_image_with_pad/pad_to_bounding_box/assert_positive/assert_less/AllZresize_image_with_pad/pad_to_bounding_box/assert_positive/assert_less/Assert/Assert/data_0*

T
2*
	summarize
�
1resize_image_with_pad/pad_to_bounding_box/Shape_1Shape+resize_image_with_pad/resize/ResizeBilinear*
T0*
out_type0*
_output_shapes
:
�
1resize_image_with_pad/pad_to_bounding_box/unstackUnpack1resize_image_with_pad/pad_to_bounding_box/Shape_1*	
num*
T0*

axis *
_output_shapes

: : : : 
r
/resize_image_with_pad/pad_to_bounding_box/sub/xConst*
value
B :�*
dtype0*
_output_shapes
: 
�
-resize_image_with_pad/pad_to_bounding_box/subSub/resize_image_with_pad/pad_to_bounding_box/sub/xresize_image_with_pad/Maximum_2*
T0*
_output_shapes
: 
�
/resize_image_with_pad/pad_to_bounding_box/sub_1Sub-resize_image_with_pad/pad_to_bounding_box/sub3resize_image_with_pad/pad_to_bounding_box/unstack:2*
_output_shapes
: *
T0
s
1resize_image_with_pad/pad_to_bounding_box/sub_2/xConst*
value	B :.*
dtype0*
_output_shapes
: 
�
/resize_image_with_pad/pad_to_bounding_box/sub_2Sub1resize_image_with_pad/pad_to_bounding_box/sub_2/xresize_image_with_pad/Maximum_1*
T0*
_output_shapes
: 
�
/resize_image_with_pad/pad_to_bounding_box/sub_3Sub/resize_image_with_pad/pad_to_bounding_box/sub_23resize_image_with_pad/pad_to_bounding_box/unstack:1*
T0*
_output_shapes
: 
z
8resize_image_with_pad/pad_to_bounding_box/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
6resize_image_with_pad/pad_to_bounding_box/GreaterEqualGreaterEqualresize_image_with_pad/Maximum_18resize_image_with_pad/pad_to_bounding_box/GreaterEqual/y*
T0*
_output_shapes
: 
�
6resize_image_with_pad/pad_to_bounding_box/Assert/ConstConst*
dtype0*
_output_shapes
: *+
value"B  Boffset_height must be >= 0
�
>resize_image_with_pad/pad_to_bounding_box/Assert/Assert/data_0Const*+
value"B  Boffset_height must be >= 0*
dtype0*
_output_shapes
: 
�
7resize_image_with_pad/pad_to_bounding_box/Assert/AssertAssert6resize_image_with_pad/pad_to_bounding_box/GreaterEqual>resize_image_with_pad/pad_to_bounding_box/Assert/Assert/data_0*

T
2*
	summarize
|
:resize_image_with_pad/pad_to_bounding_box/GreaterEqual_1/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
8resize_image_with_pad/pad_to_bounding_box/GreaterEqual_1GreaterEqualresize_image_with_pad/Maximum_2:resize_image_with_pad/pad_to_bounding_box/GreaterEqual_1/y*
T0*
_output_shapes
: 
�
8resize_image_with_pad/pad_to_bounding_box/Assert_1/ConstConst*
dtype0*
_output_shapes
: **
value!B Boffset_width must be >= 0
�
@resize_image_with_pad/pad_to_bounding_box/Assert_1/Assert/data_0Const**
value!B Boffset_width must be >= 0*
dtype0*
_output_shapes
: 
�
9resize_image_with_pad/pad_to_bounding_box/Assert_1/AssertAssert8resize_image_with_pad/pad_to_bounding_box/GreaterEqual_1@resize_image_with_pad/pad_to_bounding_box/Assert_1/Assert/data_0*

T
2*
	summarize
|
:resize_image_with_pad/pad_to_bounding_box/GreaterEqual_2/yConst*
dtype0*
_output_shapes
: *
value	B : 
�
8resize_image_with_pad/pad_to_bounding_box/GreaterEqual_2GreaterEqual/resize_image_with_pad/pad_to_bounding_box/sub_1:resize_image_with_pad/pad_to_bounding_box/GreaterEqual_2/y*
T0*
_output_shapes
: 
�
8resize_image_with_pad/pad_to_bounding_box/Assert_2/ConstConst*1
value(B& B width must be <= target - offset*
dtype0*
_output_shapes
: 
�
@resize_image_with_pad/pad_to_bounding_box/Assert_2/Assert/data_0Const*1
value(B& B width must be <= target - offset*
dtype0*
_output_shapes
: 
�
9resize_image_with_pad/pad_to_bounding_box/Assert_2/AssertAssert8resize_image_with_pad/pad_to_bounding_box/GreaterEqual_2@resize_image_with_pad/pad_to_bounding_box/Assert_2/Assert/data_0*

T
2*
	summarize
|
:resize_image_with_pad/pad_to_bounding_box/GreaterEqual_3/yConst*
value	B : *
dtype0*
_output_shapes
: 
�
8resize_image_with_pad/pad_to_bounding_box/GreaterEqual_3GreaterEqual/resize_image_with_pad/pad_to_bounding_box/sub_3:resize_image_with_pad/pad_to_bounding_box/GreaterEqual_3/y*
T0*
_output_shapes
: 
�
8resize_image_with_pad/pad_to_bounding_box/Assert_3/ConstConst*2
value)B' B!height must be <= target - offset*
dtype0*
_output_shapes
: 
�
@resize_image_with_pad/pad_to_bounding_box/Assert_3/Assert/data_0Const*2
value)B' B!height must be <= target - offset*
dtype0*
_output_shapes
: 
�
9resize_image_with_pad/pad_to_bounding_box/Assert_3/AssertAssert8resize_image_with_pad/pad_to_bounding_box/GreaterEqual_3@resize_image_with_pad/pad_to_bounding_box/Assert_3/Assert/data_0*

T
2*
	summarize
�
<resize_image_with_pad/pad_to_bounding_box/control_dependencyIdentity+resize_image_with_pad/resize/ResizeBilinear8^resize_image_with_pad/pad_to_bounding_box/Assert/Assert:^resize_image_with_pad/pad_to_bounding_box/Assert_1/Assert:^resize_image_with_pad/pad_to_bounding_box/Assert_2/Assert:^resize_image_with_pad/pad_to_bounding_box/Assert_3/AssertT^resize_image_with_pad/pad_to_bounding_box/assert_positive/assert_less/Assert/Assert*
T0*>
_class4
20loc:@resize_image_with_pad/resize/ResizeBilinear*J
_output_shapes8
6:4������������������������������������
s
1resize_image_with_pad/pad_to_bounding_box/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
s
1resize_image_with_pad/pad_to_bounding_box/stack/1Const*
value	B : *
dtype0*
_output_shapes
: 
s
1resize_image_with_pad/pad_to_bounding_box/stack/6Const*
dtype0*
_output_shapes
: *
value	B : 
s
1resize_image_with_pad/pad_to_bounding_box/stack/7Const*
value	B : *
dtype0*
_output_shapes
: 
�
/resize_image_with_pad/pad_to_bounding_box/stackPack1resize_image_with_pad/pad_to_bounding_box/stack/01resize_image_with_pad/pad_to_bounding_box/stack/1resize_image_with_pad/Maximum_1/resize_image_with_pad/pad_to_bounding_box/sub_3resize_image_with_pad/Maximum_2/resize_image_with_pad/pad_to_bounding_box/sub_11resize_image_with_pad/pad_to_bounding_box/stack/61resize_image_with_pad/pad_to_bounding_box/stack/7*
T0*

axis *
N*
_output_shapes
:
�
7resize_image_with_pad/pad_to_bounding_box/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
1resize_image_with_pad/pad_to_bounding_box/ReshapeReshape/resize_image_with_pad/pad_to_bounding_box/stack7resize_image_with_pad/pad_to_bounding_box/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
�
-resize_image_with_pad/pad_to_bounding_box/PadPad<resize_image_with_pad/pad_to_bounding_box/control_dependency1resize_image_with_pad/pad_to_bounding_box/Reshape*
T0*
	Tpaddings0*9
_output_shapes'
%:#���������.����������
�
resize_image_with_pad/Shape_2Shape-resize_image_with_pad/pad_to_bounding_box/Pad*
T0*
out_type0*
_output_shapes
:
�
resize_image_with_pad/unstack_1Unpackresize_image_with_pad/Shape_2*
_output_shapes

: : : : *	
num*
T0*

axis 
�
resize_image_with_pad/SqueezeSqueeze-resize_image_with_pad/pad_to_bounding_box/Pad*
squeeze_dims
 *
T0*,
_output_shapes
:.����������
P
ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
�

ExpandDims
ExpandDimsresize_image_with_pad/SqueezeExpandDims/dim*

Tdim0*
T0*0
_output_shapes
:.����������
�
9cnn/layer-conv1/weight/Initializer/truncated_normal/shapeConst*%
valueB"             *)
_class
loc:@cnn/layer-conv1/weight*
dtype0*
_output_shapes
:
�
8cnn/layer-conv1/weight/Initializer/truncated_normal/meanConst*
valueB
 *    *)
_class
loc:@cnn/layer-conv1/weight*
dtype0*
_output_shapes
: 
�
:cnn/layer-conv1/weight/Initializer/truncated_normal/stddevConst*
valueB
 *���=*)
_class
loc:@cnn/layer-conv1/weight*
dtype0*
_output_shapes
: 
�
Ccnn/layer-conv1/weight/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9cnn/layer-conv1/weight/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
: *

seed *
T0*)
_class
loc:@cnn/layer-conv1/weight*
seed2 
�
7cnn/layer-conv1/weight/Initializer/truncated_normal/mulMulCcnn/layer-conv1/weight/Initializer/truncated_normal/TruncatedNormal:cnn/layer-conv1/weight/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@cnn/layer-conv1/weight*&
_output_shapes
: 
�
3cnn/layer-conv1/weight/Initializer/truncated_normalAdd7cnn/layer-conv1/weight/Initializer/truncated_normal/mul8cnn/layer-conv1/weight/Initializer/truncated_normal/mean*
T0*)
_class
loc:@cnn/layer-conv1/weight*&
_output_shapes
: 
�
cnn/layer-conv1/weight
VariableV2*
dtype0*&
_output_shapes
: *
shared_name *)
_class
loc:@cnn/layer-conv1/weight*
	container *
shape: 
�
cnn/layer-conv1/weight/AssignAssigncnn/layer-conv1/weight3cnn/layer-conv1/weight/Initializer/truncated_normal*
T0*)
_class
loc:@cnn/layer-conv1/weight*
validate_shape(*&
_output_shapes
: *
use_locking(
�
cnn/layer-conv1/weight/readIdentitycnn/layer-conv1/weight*&
_output_shapes
: *
T0*)
_class
loc:@cnn/layer-conv1/weight
�
&cnn/layer-conv1/bias/Initializer/ConstConst*
valueB *    *'
_class
loc:@cnn/layer-conv1/bias*
dtype0*
_output_shapes
: 
�
cnn/layer-conv1/bias
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@cnn/layer-conv1/bias
�
cnn/layer-conv1/bias/AssignAssigncnn/layer-conv1/bias&cnn/layer-conv1/bias/Initializer/Const*
T0*'
_class
loc:@cnn/layer-conv1/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
cnn/layer-conv1/bias/readIdentitycnn/layer-conv1/bias*
_output_shapes
: *
T0*'
_class
loc:@cnn/layer-conv1/bias
�
cnn/layer-conv1/Conv2DConv2D
ExpandDimscnn/layer-conv1/weight/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:.� 
�
:cnn/layer-conv1/batch-normalization/beta/Initializer/zerosConst*
valueB *    *;
_class1
/-loc:@cnn/layer-conv1/batch-normalization/beta*
dtype0*
_output_shapes
: 
�
(cnn/layer-conv1/batch-normalization/beta
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *;
_class1
/-loc:@cnn/layer-conv1/batch-normalization/beta*
	container 
�
/cnn/layer-conv1/batch-normalization/beta/AssignAssign(cnn/layer-conv1/batch-normalization/beta:cnn/layer-conv1/batch-normalization/beta/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@cnn/layer-conv1/batch-normalization/beta*
validate_shape(*
_output_shapes
: 
�
-cnn/layer-conv1/batch-normalization/beta/readIdentity(cnn/layer-conv1/batch-normalization/beta*
T0*;
_class1
/-loc:@cnn/layer-conv1/batch-normalization/beta*
_output_shapes
: 
�
:cnn/layer-conv1/batch-normalization/gamma/Initializer/onesConst*
valueB *  �?*<
_class2
0.loc:@cnn/layer-conv1/batch-normalization/gamma*
dtype0*
_output_shapes
: 
�
)cnn/layer-conv1/batch-normalization/gamma
VariableV2*
shared_name *<
_class2
0.loc:@cnn/layer-conv1/batch-normalization/gamma*
	container *
shape: *
dtype0*
_output_shapes
: 
�
0cnn/layer-conv1/batch-normalization/gamma/AssignAssign)cnn/layer-conv1/batch-normalization/gamma:cnn/layer-conv1/batch-normalization/gamma/Initializer/ones*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*<
_class2
0.loc:@cnn/layer-conv1/batch-normalization/gamma
�
.cnn/layer-conv1/batch-normalization/gamma/readIdentity)cnn/layer-conv1/batch-normalization/gamma*
T0*<
_class2
0.loc:@cnn/layer-conv1/batch-normalization/gamma*
_output_shapes
: 
�
Acnn/layer-conv1/batch-normalization/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *B
_class8
64loc:@cnn/layer-conv1/batch-normalization/moving_mean
�
/cnn/layer-conv1/batch-normalization/moving_mean
VariableV2*
dtype0*
_output_shapes
: *
shared_name *B
_class8
64loc:@cnn/layer-conv1/batch-normalization/moving_mean*
	container *
shape: 
�
6cnn/layer-conv1/batch-normalization/moving_mean/AssignAssign/cnn/layer-conv1/batch-normalization/moving_meanAcnn/layer-conv1/batch-normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@cnn/layer-conv1/batch-normalization/moving_mean*
validate_shape(*
_output_shapes
: 
�
4cnn/layer-conv1/batch-normalization/moving_mean/readIdentity/cnn/layer-conv1/batch-normalization/moving_mean*
T0*B
_class8
64loc:@cnn/layer-conv1/batch-normalization/moving_mean*
_output_shapes
: 
�
Dcnn/layer-conv1/batch-normalization/moving_variance/Initializer/onesConst*
valueB *  �?*F
_class<
:8loc:@cnn/layer-conv1/batch-normalization/moving_variance*
dtype0*
_output_shapes
: 
�
3cnn/layer-conv1/batch-normalization/moving_variance
VariableV2*F
_class<
:8loc:@cnn/layer-conv1/batch-normalization/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
:cnn/layer-conv1/batch-normalization/moving_variance/AssignAssign3cnn/layer-conv1/batch-normalization/moving_varianceDcnn/layer-conv1/batch-normalization/moving_variance/Initializer/ones*
use_locking(*
T0*F
_class<
:8loc:@cnn/layer-conv1/batch-normalization/moving_variance*
validate_shape(*
_output_shapes
: 
�
8cnn/layer-conv1/batch-normalization/moving_variance/readIdentity3cnn/layer-conv1/batch-normalization/moving_variance*
T0*F
_class<
:8loc:@cnn/layer-conv1/batch-normalization/moving_variance*
_output_shapes
: 
�
2cnn/layer-conv1/batch-normalization/FusedBatchNormFusedBatchNormcnn/layer-conv1/Conv2D.cnn/layer-conv1/batch-normalization/gamma/read-cnn/layer-conv1/batch-normalization/beta/read4cnn/layer-conv1/batch-normalization/moving_mean/read8cnn/layer-conv1/batch-normalization/moving_variance/read*
data_formatNHWC*?
_output_shapes-
+:.� : : : : *
is_training( *
epsilon%��'7*
T0
�
cnn/layer-conv1/BiasAddBiasAdd2cnn/layer-conv1/batch-normalization/FusedBatchNormcnn/layer-conv1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:.� 
g
cnn/layer-conv1/ReluRelucnn/layer-conv1/BiasAdd*
T0*'
_output_shapes
:.� 
�
;cnn/layer-conv1-1/weight/Initializer/truncated_normal/shapeConst*%
valueB"              *+
_class!
loc:@cnn/layer-conv1-1/weight*
dtype0*
_output_shapes
:
�
:cnn/layer-conv1-1/weight/Initializer/truncated_normal/meanConst*
valueB
 *    *+
_class!
loc:@cnn/layer-conv1-1/weight*
dtype0*
_output_shapes
: 
�
<cnn/layer-conv1-1/weight/Initializer/truncated_normal/stddevConst*
valueB
 *���=*+
_class!
loc:@cnn/layer-conv1-1/weight*
dtype0*
_output_shapes
: 
�
Ecnn/layer-conv1-1/weight/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;cnn/layer-conv1-1/weight/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:  *

seed *
T0*+
_class!
loc:@cnn/layer-conv1-1/weight*
seed2 
�
9cnn/layer-conv1-1/weight/Initializer/truncated_normal/mulMulEcnn/layer-conv1-1/weight/Initializer/truncated_normal/TruncatedNormal<cnn/layer-conv1-1/weight/Initializer/truncated_normal/stddev*
T0*+
_class!
loc:@cnn/layer-conv1-1/weight*&
_output_shapes
:  
�
5cnn/layer-conv1-1/weight/Initializer/truncated_normalAdd9cnn/layer-conv1-1/weight/Initializer/truncated_normal/mul:cnn/layer-conv1-1/weight/Initializer/truncated_normal/mean*
T0*+
_class!
loc:@cnn/layer-conv1-1/weight*&
_output_shapes
:  
�
cnn/layer-conv1-1/weight
VariableV2*
shape:  *
dtype0*&
_output_shapes
:  *
shared_name *+
_class!
loc:@cnn/layer-conv1-1/weight*
	container 
�
cnn/layer-conv1-1/weight/AssignAssigncnn/layer-conv1-1/weight5cnn/layer-conv1-1/weight/Initializer/truncated_normal*
use_locking(*
T0*+
_class!
loc:@cnn/layer-conv1-1/weight*
validate_shape(*&
_output_shapes
:  
�
cnn/layer-conv1-1/weight/readIdentitycnn/layer-conv1-1/weight*&
_output_shapes
:  *
T0*+
_class!
loc:@cnn/layer-conv1-1/weight
�
(cnn/layer-conv1-1/bias/Initializer/ConstConst*
dtype0*
_output_shapes
: *
valueB *    *)
_class
loc:@cnn/layer-conv1-1/bias
�
cnn/layer-conv1-1/bias
VariableV2*
dtype0*
_output_shapes
: *
shared_name *)
_class
loc:@cnn/layer-conv1-1/bias*
	container *
shape: 
�
cnn/layer-conv1-1/bias/AssignAssigncnn/layer-conv1-1/bias(cnn/layer-conv1-1/bias/Initializer/Const*
use_locking(*
T0*)
_class
loc:@cnn/layer-conv1-1/bias*
validate_shape(*
_output_shapes
: 
�
cnn/layer-conv1-1/bias/readIdentitycnn/layer-conv1-1/bias*
T0*)
_class
loc:@cnn/layer-conv1-1/bias*
_output_shapes
: 
�
cnn/layer-conv1-1/Conv2DConv2Dcnn/layer-conv1/Relucnn/layer-conv1-1/weight/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:.� 
�
<cnn/layer-conv1-1/batch-normalization/beta/Initializer/zerosConst*
valueB *    *=
_class3
1/loc:@cnn/layer-conv1-1/batch-normalization/beta*
dtype0*
_output_shapes
: 
�
*cnn/layer-conv1-1/batch-normalization/beta
VariableV2*
shared_name *=
_class3
1/loc:@cnn/layer-conv1-1/batch-normalization/beta*
	container *
shape: *
dtype0*
_output_shapes
: 
�
1cnn/layer-conv1-1/batch-normalization/beta/AssignAssign*cnn/layer-conv1-1/batch-normalization/beta<cnn/layer-conv1-1/batch-normalization/beta/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@cnn/layer-conv1-1/batch-normalization/beta*
validate_shape(*
_output_shapes
: 
�
/cnn/layer-conv1-1/batch-normalization/beta/readIdentity*cnn/layer-conv1-1/batch-normalization/beta*
T0*=
_class3
1/loc:@cnn/layer-conv1-1/batch-normalization/beta*
_output_shapes
: 
�
<cnn/layer-conv1-1/batch-normalization/gamma/Initializer/onesConst*
valueB *  �?*>
_class4
20loc:@cnn/layer-conv1-1/batch-normalization/gamma*
dtype0*
_output_shapes
: 
�
+cnn/layer-conv1-1/batch-normalization/gamma
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *>
_class4
20loc:@cnn/layer-conv1-1/batch-normalization/gamma*
	container 
�
2cnn/layer-conv1-1/batch-normalization/gamma/AssignAssign+cnn/layer-conv1-1/batch-normalization/gamma<cnn/layer-conv1-1/batch-normalization/gamma/Initializer/ones*
T0*>
_class4
20loc:@cnn/layer-conv1-1/batch-normalization/gamma*
validate_shape(*
_output_shapes
: *
use_locking(
�
0cnn/layer-conv1-1/batch-normalization/gamma/readIdentity+cnn/layer-conv1-1/batch-normalization/gamma*
T0*>
_class4
20loc:@cnn/layer-conv1-1/batch-normalization/gamma*
_output_shapes
: 
�
Ccnn/layer-conv1-1/batch-normalization/moving_mean/Initializer/zerosConst*
valueB *    *D
_class:
86loc:@cnn/layer-conv1-1/batch-normalization/moving_mean*
dtype0*
_output_shapes
: 
�
1cnn/layer-conv1-1/batch-normalization/moving_mean
VariableV2*
shared_name *D
_class:
86loc:@cnn/layer-conv1-1/batch-normalization/moving_mean*
	container *
shape: *
dtype0*
_output_shapes
: 
�
8cnn/layer-conv1-1/batch-normalization/moving_mean/AssignAssign1cnn/layer-conv1-1/batch-normalization/moving_meanCcnn/layer-conv1-1/batch-normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@cnn/layer-conv1-1/batch-normalization/moving_mean*
validate_shape(*
_output_shapes
: 
�
6cnn/layer-conv1-1/batch-normalization/moving_mean/readIdentity1cnn/layer-conv1-1/batch-normalization/moving_mean*
T0*D
_class:
86loc:@cnn/layer-conv1-1/batch-normalization/moving_mean*
_output_shapes
: 
�
Fcnn/layer-conv1-1/batch-normalization/moving_variance/Initializer/onesConst*
valueB *  �?*H
_class>
<:loc:@cnn/layer-conv1-1/batch-normalization/moving_variance*
dtype0*
_output_shapes
: 
�
5cnn/layer-conv1-1/batch-normalization/moving_variance
VariableV2*
dtype0*
_output_shapes
: *
shared_name *H
_class>
<:loc:@cnn/layer-conv1-1/batch-normalization/moving_variance*
	container *
shape: 
�
<cnn/layer-conv1-1/batch-normalization/moving_variance/AssignAssign5cnn/layer-conv1-1/batch-normalization/moving_varianceFcnn/layer-conv1-1/batch-normalization/moving_variance/Initializer/ones*
T0*H
_class>
<:loc:@cnn/layer-conv1-1/batch-normalization/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
�
:cnn/layer-conv1-1/batch-normalization/moving_variance/readIdentity5cnn/layer-conv1-1/batch-normalization/moving_variance*
T0*H
_class>
<:loc:@cnn/layer-conv1-1/batch-normalization/moving_variance*
_output_shapes
: 
�
4cnn/layer-conv1-1/batch-normalization/FusedBatchNormFusedBatchNormcnn/layer-conv1-1/Conv2D0cnn/layer-conv1-1/batch-normalization/gamma/read/cnn/layer-conv1-1/batch-normalization/beta/read6cnn/layer-conv1-1/batch-normalization/moving_mean/read:cnn/layer-conv1-1/batch-normalization/moving_variance/read*
epsilon%��'7*
T0*
data_formatNHWC*?
_output_shapes-
+:.� : : : : *
is_training( 
�
cnn/layer-conv1-1/BiasAddBiasAdd4cnn/layer-conv1-1/batch-normalization/FusedBatchNormcnn/layer-conv1-1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:.� 
k
cnn/layer-conv1-1/ReluRelucnn/layer-conv1-1/BiasAdd*
T0*'
_output_shapes
:.� 
�
cnn/pooling-layer-1/MaxPoolMaxPoolcnn/layer-conv1-1/Relu*'
_output_shapes
:� *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
�
9cnn/layer-conv2/weight/Initializer/truncated_normal/shapeConst*%
valueB"          @   *)
_class
loc:@cnn/layer-conv2/weight*
dtype0*
_output_shapes
:
�
8cnn/layer-conv2/weight/Initializer/truncated_normal/meanConst*
valueB
 *    *)
_class
loc:@cnn/layer-conv2/weight*
dtype0*
_output_shapes
: 
�
:cnn/layer-conv2/weight/Initializer/truncated_normal/stddevConst*
valueB
 *���=*)
_class
loc:@cnn/layer-conv2/weight*
dtype0*
_output_shapes
: 
�
Ccnn/layer-conv2/weight/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9cnn/layer-conv2/weight/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
: @*

seed *
T0*)
_class
loc:@cnn/layer-conv2/weight*
seed2 
�
7cnn/layer-conv2/weight/Initializer/truncated_normal/mulMulCcnn/layer-conv2/weight/Initializer/truncated_normal/TruncatedNormal:cnn/layer-conv2/weight/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@cnn/layer-conv2/weight*&
_output_shapes
: @
�
3cnn/layer-conv2/weight/Initializer/truncated_normalAdd7cnn/layer-conv2/weight/Initializer/truncated_normal/mul8cnn/layer-conv2/weight/Initializer/truncated_normal/mean*&
_output_shapes
: @*
T0*)
_class
loc:@cnn/layer-conv2/weight
�
cnn/layer-conv2/weight
VariableV2*
shape: @*
dtype0*&
_output_shapes
: @*
shared_name *)
_class
loc:@cnn/layer-conv2/weight*
	container 
�
cnn/layer-conv2/weight/AssignAssigncnn/layer-conv2/weight3cnn/layer-conv2/weight/Initializer/truncated_normal*
use_locking(*
T0*)
_class
loc:@cnn/layer-conv2/weight*
validate_shape(*&
_output_shapes
: @
�
cnn/layer-conv2/weight/readIdentitycnn/layer-conv2/weight*
T0*)
_class
loc:@cnn/layer-conv2/weight*&
_output_shapes
: @
�
&cnn/layer-conv2/bias/Initializer/ConstConst*
valueB@*    *'
_class
loc:@cnn/layer-conv2/bias*
dtype0*
_output_shapes
:@
�
cnn/layer-conv2/bias
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@cnn/layer-conv2/bias*
	container *
shape:@
�
cnn/layer-conv2/bias/AssignAssigncnn/layer-conv2/bias&cnn/layer-conv2/bias/Initializer/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@cnn/layer-conv2/bias
�
cnn/layer-conv2/bias/readIdentitycnn/layer-conv2/bias*
_output_shapes
:@*
T0*'
_class
loc:@cnn/layer-conv2/bias
�
cnn/layer-conv2/Conv2DConv2Dcnn/pooling-layer-1/MaxPoolcnn/layer-conv2/weight/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:�@*
	dilations
*
T0
�
:cnn/layer-conv2/batch-normalization/beta/Initializer/zerosConst*
valueB@*    *;
_class1
/-loc:@cnn/layer-conv2/batch-normalization/beta*
dtype0*
_output_shapes
:@
�
(cnn/layer-conv2/batch-normalization/beta
VariableV2*
shared_name *;
_class1
/-loc:@cnn/layer-conv2/batch-normalization/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
/cnn/layer-conv2/batch-normalization/beta/AssignAssign(cnn/layer-conv2/batch-normalization/beta:cnn/layer-conv2/batch-normalization/beta/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@cnn/layer-conv2/batch-normalization/beta*
validate_shape(*
_output_shapes
:@
�
-cnn/layer-conv2/batch-normalization/beta/readIdentity(cnn/layer-conv2/batch-normalization/beta*
T0*;
_class1
/-loc:@cnn/layer-conv2/batch-normalization/beta*
_output_shapes
:@
�
:cnn/layer-conv2/batch-normalization/gamma/Initializer/onesConst*
valueB@*  �?*<
_class2
0.loc:@cnn/layer-conv2/batch-normalization/gamma*
dtype0*
_output_shapes
:@
�
)cnn/layer-conv2/batch-normalization/gamma
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *<
_class2
0.loc:@cnn/layer-conv2/batch-normalization/gamma
�
0cnn/layer-conv2/batch-normalization/gamma/AssignAssign)cnn/layer-conv2/batch-normalization/gamma:cnn/layer-conv2/batch-normalization/gamma/Initializer/ones*
use_locking(*
T0*<
_class2
0.loc:@cnn/layer-conv2/batch-normalization/gamma*
validate_shape(*
_output_shapes
:@
�
.cnn/layer-conv2/batch-normalization/gamma/readIdentity)cnn/layer-conv2/batch-normalization/gamma*
T0*<
_class2
0.loc:@cnn/layer-conv2/batch-normalization/gamma*
_output_shapes
:@
�
Acnn/layer-conv2/batch-normalization/moving_mean/Initializer/zerosConst*
valueB@*    *B
_class8
64loc:@cnn/layer-conv2/batch-normalization/moving_mean*
dtype0*
_output_shapes
:@
�
/cnn/layer-conv2/batch-normalization/moving_mean
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *B
_class8
64loc:@cnn/layer-conv2/batch-normalization/moving_mean*
	container 
�
6cnn/layer-conv2/batch-normalization/moving_mean/AssignAssign/cnn/layer-conv2/batch-normalization/moving_meanAcnn/layer-conv2/batch-normalization/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*B
_class8
64loc:@cnn/layer-conv2/batch-normalization/moving_mean
�
4cnn/layer-conv2/batch-normalization/moving_mean/readIdentity/cnn/layer-conv2/batch-normalization/moving_mean*
T0*B
_class8
64loc:@cnn/layer-conv2/batch-normalization/moving_mean*
_output_shapes
:@
�
Dcnn/layer-conv2/batch-normalization/moving_variance/Initializer/onesConst*
valueB@*  �?*F
_class<
:8loc:@cnn/layer-conv2/batch-normalization/moving_variance*
dtype0*
_output_shapes
:@
�
3cnn/layer-conv2/batch-normalization/moving_variance
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *F
_class<
:8loc:@cnn/layer-conv2/batch-normalization/moving_variance*
	container *
shape:@
�
:cnn/layer-conv2/batch-normalization/moving_variance/AssignAssign3cnn/layer-conv2/batch-normalization/moving_varianceDcnn/layer-conv2/batch-normalization/moving_variance/Initializer/ones*
T0*F
_class<
:8loc:@cnn/layer-conv2/batch-normalization/moving_variance*
validate_shape(*
_output_shapes
:@*
use_locking(
�
8cnn/layer-conv2/batch-normalization/moving_variance/readIdentity3cnn/layer-conv2/batch-normalization/moving_variance*
T0*F
_class<
:8loc:@cnn/layer-conv2/batch-normalization/moving_variance*
_output_shapes
:@
�
2cnn/layer-conv2/batch-normalization/FusedBatchNormFusedBatchNormcnn/layer-conv2/Conv2D.cnn/layer-conv2/batch-normalization/gamma/read-cnn/layer-conv2/batch-normalization/beta/read4cnn/layer-conv2/batch-normalization/moving_mean/read8cnn/layer-conv2/batch-normalization/moving_variance/read*
epsilon%��'7*
T0*
data_formatNHWC*?
_output_shapes-
+:�@:@:@:@:@*
is_training( 
�
cnn/layer-conv2/BiasAddBiasAdd2cnn/layer-conv2/batch-normalization/FusedBatchNormcnn/layer-conv2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:�@
g
cnn/layer-conv2/ReluRelucnn/layer-conv2/BiasAdd*
T0*'
_output_shapes
:�@
�
cnn/pooling-layer-2/MaxPoolMaxPoolcnn/layer-conv2/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*&
_output_shapes
:}@*
T0
�
9cnn/layer-conv3/weight/Initializer/truncated_normal/shapeConst*%
valueB"      @   @   *)
_class
loc:@cnn/layer-conv3/weight*
dtype0*
_output_shapes
:
�
8cnn/layer-conv3/weight/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *)
_class
loc:@cnn/layer-conv3/weight
�
:cnn/layer-conv3/weight/Initializer/truncated_normal/stddevConst*
valueB
 *���=*)
_class
loc:@cnn/layer-conv3/weight*
dtype0*
_output_shapes
: 
�
Ccnn/layer-conv3/weight/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9cnn/layer-conv3/weight/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*)
_class
loc:@cnn/layer-conv3/weight*
seed2 
�
7cnn/layer-conv3/weight/Initializer/truncated_normal/mulMulCcnn/layer-conv3/weight/Initializer/truncated_normal/TruncatedNormal:cnn/layer-conv3/weight/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@cnn/layer-conv3/weight*&
_output_shapes
:@@
�
3cnn/layer-conv3/weight/Initializer/truncated_normalAdd7cnn/layer-conv3/weight/Initializer/truncated_normal/mul8cnn/layer-conv3/weight/Initializer/truncated_normal/mean*&
_output_shapes
:@@*
T0*)
_class
loc:@cnn/layer-conv3/weight
�
cnn/layer-conv3/weight
VariableV2*)
_class
loc:@cnn/layer-conv3/weight*
	container *
shape:@@*
dtype0*&
_output_shapes
:@@*
shared_name 
�
cnn/layer-conv3/weight/AssignAssigncnn/layer-conv3/weight3cnn/layer-conv3/weight/Initializer/truncated_normal*
use_locking(*
T0*)
_class
loc:@cnn/layer-conv3/weight*
validate_shape(*&
_output_shapes
:@@
�
cnn/layer-conv3/weight/readIdentitycnn/layer-conv3/weight*
T0*)
_class
loc:@cnn/layer-conv3/weight*&
_output_shapes
:@@
�
&cnn/layer-conv3/bias/Initializer/ConstConst*
valueB@*    *'
_class
loc:@cnn/layer-conv3/bias*
dtype0*
_output_shapes
:@
�
cnn/layer-conv3/bias
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@cnn/layer-conv3/bias
�
cnn/layer-conv3/bias/AssignAssigncnn/layer-conv3/bias&cnn/layer-conv3/bias/Initializer/Const*
use_locking(*
T0*'
_class
loc:@cnn/layer-conv3/bias*
validate_shape(*
_output_shapes
:@
�
cnn/layer-conv3/bias/readIdentitycnn/layer-conv3/bias*
T0*'
_class
loc:@cnn/layer-conv3/bias*
_output_shapes
:@
�
cnn/layer-conv3/Conv2DConv2Dcnn/pooling-layer-2/MaxPoolcnn/layer-conv3/weight/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:}@
�
:cnn/layer-conv3/batch-normalization/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *;
_class1
/-loc:@cnn/layer-conv3/batch-normalization/beta
�
(cnn/layer-conv3/batch-normalization/beta
VariableV2*
shared_name *;
_class1
/-loc:@cnn/layer-conv3/batch-normalization/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
/cnn/layer-conv3/batch-normalization/beta/AssignAssign(cnn/layer-conv3/batch-normalization/beta:cnn/layer-conv3/batch-normalization/beta/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@cnn/layer-conv3/batch-normalization/beta*
validate_shape(*
_output_shapes
:@
�
-cnn/layer-conv3/batch-normalization/beta/readIdentity(cnn/layer-conv3/batch-normalization/beta*
T0*;
_class1
/-loc:@cnn/layer-conv3/batch-normalization/beta*
_output_shapes
:@
�
:cnn/layer-conv3/batch-normalization/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:@*
valueB@*  �?*<
_class2
0.loc:@cnn/layer-conv3/batch-normalization/gamma
�
)cnn/layer-conv3/batch-normalization/gamma
VariableV2*
shared_name *<
_class2
0.loc:@cnn/layer-conv3/batch-normalization/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
0cnn/layer-conv3/batch-normalization/gamma/AssignAssign)cnn/layer-conv3/batch-normalization/gamma:cnn/layer-conv3/batch-normalization/gamma/Initializer/ones*
use_locking(*
T0*<
_class2
0.loc:@cnn/layer-conv3/batch-normalization/gamma*
validate_shape(*
_output_shapes
:@
�
.cnn/layer-conv3/batch-normalization/gamma/readIdentity)cnn/layer-conv3/batch-normalization/gamma*
T0*<
_class2
0.loc:@cnn/layer-conv3/batch-normalization/gamma*
_output_shapes
:@
�
Acnn/layer-conv3/batch-normalization/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *B
_class8
64loc:@cnn/layer-conv3/batch-normalization/moving_mean
�
/cnn/layer-conv3/batch-normalization/moving_mean
VariableV2*
shared_name *B
_class8
64loc:@cnn/layer-conv3/batch-normalization/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
6cnn/layer-conv3/batch-normalization/moving_mean/AssignAssign/cnn/layer-conv3/batch-normalization/moving_meanAcnn/layer-conv3/batch-normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@cnn/layer-conv3/batch-normalization/moving_mean*
validate_shape(*
_output_shapes
:@
�
4cnn/layer-conv3/batch-normalization/moving_mean/readIdentity/cnn/layer-conv3/batch-normalization/moving_mean*
T0*B
_class8
64loc:@cnn/layer-conv3/batch-normalization/moving_mean*
_output_shapes
:@
�
Dcnn/layer-conv3/batch-normalization/moving_variance/Initializer/onesConst*
valueB@*  �?*F
_class<
:8loc:@cnn/layer-conv3/batch-normalization/moving_variance*
dtype0*
_output_shapes
:@
�
3cnn/layer-conv3/batch-normalization/moving_variance
VariableV2*
shared_name *F
_class<
:8loc:@cnn/layer-conv3/batch-normalization/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
:cnn/layer-conv3/batch-normalization/moving_variance/AssignAssign3cnn/layer-conv3/batch-normalization/moving_varianceDcnn/layer-conv3/batch-normalization/moving_variance/Initializer/ones*
use_locking(*
T0*F
_class<
:8loc:@cnn/layer-conv3/batch-normalization/moving_variance*
validate_shape(*
_output_shapes
:@
�
8cnn/layer-conv3/batch-normalization/moving_variance/readIdentity3cnn/layer-conv3/batch-normalization/moving_variance*
_output_shapes
:@*
T0*F
_class<
:8loc:@cnn/layer-conv3/batch-normalization/moving_variance
�
2cnn/layer-conv3/batch-normalization/FusedBatchNormFusedBatchNormcnn/layer-conv3/Conv2D.cnn/layer-conv3/batch-normalization/gamma/read-cnn/layer-conv3/batch-normalization/beta/read4cnn/layer-conv3/batch-normalization/moving_mean/read8cnn/layer-conv3/batch-normalization/moving_variance/read*
T0*
data_formatNHWC*>
_output_shapes,
*:}@:@:@:@:@*
is_training( *
epsilon%��'7
�
cnn/layer-conv3/BiasAddBiasAdd2cnn/layer-conv3/batch-normalization/FusedBatchNormcnn/layer-conv3/bias/read*
T0*
data_formatNHWC*&
_output_shapes
:}@
f
cnn/layer-conv3/ReluRelucnn/layer-conv3/BiasAdd*
T0*&
_output_shapes
:}@
�
;cnn/layer-conv3-1/weight/Initializer/truncated_normal/shapeConst*%
valueB"      @   @   *+
_class!
loc:@cnn/layer-conv3-1/weight*
dtype0*
_output_shapes
:
�
:cnn/layer-conv3-1/weight/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *+
_class!
loc:@cnn/layer-conv3-1/weight
�
<cnn/layer-conv3-1/weight/Initializer/truncated_normal/stddevConst*
valueB
 *���=*+
_class!
loc:@cnn/layer-conv3-1/weight*
dtype0*
_output_shapes
: 
�
Ecnn/layer-conv3-1/weight/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;cnn/layer-conv3-1/weight/Initializer/truncated_normal/shape*
T0*+
_class!
loc:@cnn/layer-conv3-1/weight*
seed2 *
dtype0*&
_output_shapes
:@@*

seed 
�
9cnn/layer-conv3-1/weight/Initializer/truncated_normal/mulMulEcnn/layer-conv3-1/weight/Initializer/truncated_normal/TruncatedNormal<cnn/layer-conv3-1/weight/Initializer/truncated_normal/stddev*
T0*+
_class!
loc:@cnn/layer-conv3-1/weight*&
_output_shapes
:@@
�
5cnn/layer-conv3-1/weight/Initializer/truncated_normalAdd9cnn/layer-conv3-1/weight/Initializer/truncated_normal/mul:cnn/layer-conv3-1/weight/Initializer/truncated_normal/mean*&
_output_shapes
:@@*
T0*+
_class!
loc:@cnn/layer-conv3-1/weight
�
cnn/layer-conv3-1/weight
VariableV2*
shape:@@*
dtype0*&
_output_shapes
:@@*
shared_name *+
_class!
loc:@cnn/layer-conv3-1/weight*
	container 
�
cnn/layer-conv3-1/weight/AssignAssigncnn/layer-conv3-1/weight5cnn/layer-conv3-1/weight/Initializer/truncated_normal*
use_locking(*
T0*+
_class!
loc:@cnn/layer-conv3-1/weight*
validate_shape(*&
_output_shapes
:@@
�
cnn/layer-conv3-1/weight/readIdentitycnn/layer-conv3-1/weight*
T0*+
_class!
loc:@cnn/layer-conv3-1/weight*&
_output_shapes
:@@
�
(cnn/layer-conv3-1/bias/Initializer/ConstConst*
valueB@*    *)
_class
loc:@cnn/layer-conv3-1/bias*
dtype0*
_output_shapes
:@
�
cnn/layer-conv3-1/bias
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *)
_class
loc:@cnn/layer-conv3-1/bias
�
cnn/layer-conv3-1/bias/AssignAssigncnn/layer-conv3-1/bias(cnn/layer-conv3-1/bias/Initializer/Const*
use_locking(*
T0*)
_class
loc:@cnn/layer-conv3-1/bias*
validate_shape(*
_output_shapes
:@
�
cnn/layer-conv3-1/bias/readIdentitycnn/layer-conv3-1/bias*
_output_shapes
:@*
T0*)
_class
loc:@cnn/layer-conv3-1/bias
�
cnn/layer-conv3-1/Conv2DConv2Dcnn/layer-conv3/Relucnn/layer-conv3-1/weight/read*&
_output_shapes
:}@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
<cnn/layer-conv3-1/batch-normalization/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *=
_class3
1/loc:@cnn/layer-conv3-1/batch-normalization/beta
�
*cnn/layer-conv3-1/batch-normalization/beta
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *=
_class3
1/loc:@cnn/layer-conv3-1/batch-normalization/beta*
	container *
shape:@
�
1cnn/layer-conv3-1/batch-normalization/beta/AssignAssign*cnn/layer-conv3-1/batch-normalization/beta<cnn/layer-conv3-1/batch-normalization/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*=
_class3
1/loc:@cnn/layer-conv3-1/batch-normalization/beta
�
/cnn/layer-conv3-1/batch-normalization/beta/readIdentity*cnn/layer-conv3-1/batch-normalization/beta*
T0*=
_class3
1/loc:@cnn/layer-conv3-1/batch-normalization/beta*
_output_shapes
:@
�
<cnn/layer-conv3-1/batch-normalization/gamma/Initializer/onesConst*
valueB@*  �?*>
_class4
20loc:@cnn/layer-conv3-1/batch-normalization/gamma*
dtype0*
_output_shapes
:@
�
+cnn/layer-conv3-1/batch-normalization/gamma
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *>
_class4
20loc:@cnn/layer-conv3-1/batch-normalization/gamma*
	container *
shape:@
�
2cnn/layer-conv3-1/batch-normalization/gamma/AssignAssign+cnn/layer-conv3-1/batch-normalization/gamma<cnn/layer-conv3-1/batch-normalization/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*>
_class4
20loc:@cnn/layer-conv3-1/batch-normalization/gamma
�
0cnn/layer-conv3-1/batch-normalization/gamma/readIdentity+cnn/layer-conv3-1/batch-normalization/gamma*
T0*>
_class4
20loc:@cnn/layer-conv3-1/batch-normalization/gamma*
_output_shapes
:@
�
Ccnn/layer-conv3-1/batch-normalization/moving_mean/Initializer/zerosConst*
valueB@*    *D
_class:
86loc:@cnn/layer-conv3-1/batch-normalization/moving_mean*
dtype0*
_output_shapes
:@
�
1cnn/layer-conv3-1/batch-normalization/moving_mean
VariableV2*
shared_name *D
_class:
86loc:@cnn/layer-conv3-1/batch-normalization/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
8cnn/layer-conv3-1/batch-normalization/moving_mean/AssignAssign1cnn/layer-conv3-1/batch-normalization/moving_meanCcnn/layer-conv3-1/batch-normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@cnn/layer-conv3-1/batch-normalization/moving_mean*
validate_shape(*
_output_shapes
:@
�
6cnn/layer-conv3-1/batch-normalization/moving_mean/readIdentity1cnn/layer-conv3-1/batch-normalization/moving_mean*
T0*D
_class:
86loc:@cnn/layer-conv3-1/batch-normalization/moving_mean*
_output_shapes
:@
�
Fcnn/layer-conv3-1/batch-normalization/moving_variance/Initializer/onesConst*
valueB@*  �?*H
_class>
<:loc:@cnn/layer-conv3-1/batch-normalization/moving_variance*
dtype0*
_output_shapes
:@
�
5cnn/layer-conv3-1/batch-normalization/moving_variance
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *H
_class>
<:loc:@cnn/layer-conv3-1/batch-normalization/moving_variance*
	container *
shape:@
�
<cnn/layer-conv3-1/batch-normalization/moving_variance/AssignAssign5cnn/layer-conv3-1/batch-normalization/moving_varianceFcnn/layer-conv3-1/batch-normalization/moving_variance/Initializer/ones*
use_locking(*
T0*H
_class>
<:loc:@cnn/layer-conv3-1/batch-normalization/moving_variance*
validate_shape(*
_output_shapes
:@
�
:cnn/layer-conv3-1/batch-normalization/moving_variance/readIdentity5cnn/layer-conv3-1/batch-normalization/moving_variance*
T0*H
_class>
<:loc:@cnn/layer-conv3-1/batch-normalization/moving_variance*
_output_shapes
:@
�
4cnn/layer-conv3-1/batch-normalization/FusedBatchNormFusedBatchNormcnn/layer-conv3-1/Conv2D0cnn/layer-conv3-1/batch-normalization/gamma/read/cnn/layer-conv3-1/batch-normalization/beta/read6cnn/layer-conv3-1/batch-normalization/moving_mean/read:cnn/layer-conv3-1/batch-normalization/moving_variance/read*
epsilon%��'7*
T0*
data_formatNHWC*>
_output_shapes,
*:}@:@:@:@:@*
is_training( 
�
cnn/layer-conv3-1/BiasAddBiasAdd4cnn/layer-conv3-1/batch-normalization/FusedBatchNormcnn/layer-conv3-1/bias/read*
T0*
data_formatNHWC*&
_output_shapes
:}@
j
cnn/layer-conv3-1/ReluRelucnn/layer-conv3-1/BiasAdd*
T0*&
_output_shapes
:}@
S
	Fill/dimsConst*
valueB:*
dtype0*
_output_shapes
:
L

Fill/valueConst*
value	B :}*
dtype0*
_output_shapes
: 
Z
FillFill	Fill/dims
Fill/value*
T0*

index_type0*
_output_shapes
:
g
transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
|
	transpose	Transposecnn/layer-conv3-1/Relutranspose/perm*
Tperm0*
T0*&
_output_shapes
:}@
b
Reshape/shapeConst*!
valueB"����}      *
dtype0*
_output_shapes
:
h
ReshapeReshape	transposeReshape/shape*
T0*
Tshape0*#
_output_shapes
:}�
|
2lstm/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

4lstm/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:�*
dtype0*
_output_shapes
:
z
8lstm/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
3lstm/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV22lstm/MultiRNNCellZeroState/LSTMCellZeroState/Const4lstm/MultiRNNCellZeroState/LSTMCellZeroState/Const_18lstm/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
}
8lstm/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
2lstm/MultiRNNCellZeroState/LSTMCellZeroState/zerosFill3lstm/MultiRNNCellZeroState/LSTMCellZeroState/concat8lstm/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
_output_shapes
:	�*
T0*

index_type0
~
4lstm/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

4lstm/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:�*
dtype0*
_output_shapes
:
~
4lstm/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

4lstm/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
dtype0*
_output_shapes
:*
valueB:�
|
:lstm/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
5lstm/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV24lstm/MultiRNNCellZeroState/LSTMCellZeroState/Const_44lstm/MultiRNNCellZeroState/LSTMCellZeroState/Const_5:lstm/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:

:lstm/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
4lstm/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1Fill5lstm/MultiRNNCellZeroState/LSTMCellZeroState/concat_1:lstm/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0*
_output_shapes
:	�
~
4lstm/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

4lstm/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:�*
dtype0*
_output_shapes
:
~
4lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
dtype0*
_output_shapes
:*
valueB:
�
6lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:�*
dtype0*
_output_shapes
:
|
:lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
5lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV24lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/Const6lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1:lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

:lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
4lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFill5lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/concat:lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*

index_type0*
_output_shapes
:	�
�
6lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
�
6lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:�*
dtype0*
_output_shapes
:
�
6lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
dtype0*
_output_shapes
:*
valueB:
�
6lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
dtype0*
_output_shapes
:*
valueB:�
~
<lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
7lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV26lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_46lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5<lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
<lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1Fill7lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1<lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
_output_shapes
:	�*
T0*

index_type0
�
6lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
�
6lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:�*
dtype0*
_output_shapes
:
O
lstm/rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
V
lstm/rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
V
lstm/rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
z
lstm/rnn/rangeRangelstm/rnn/range/startlstm/rnn/Ranklstm/rnn/range/delta*
_output_shapes
:*

Tidx0
i
lstm/rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
V
lstm/rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
lstm/rnn/concatConcatV2lstm/rnn/concat/values_0lstm/rnn/rangelstm/rnn/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
t
lstm/rnn/transpose	TransposeReshapelstm/rnn/concat*#
_output_shapes
:}�*
Tperm0*
T0
O
lstm/rnn/sequence_lengthIdentityFill*
T0*
_output_shapes
:
X
lstm/rnn/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
X
lstm/rnn/stackConst*
valueB:*
dtype0*
_output_shapes
:
\
lstm/rnn/EqualEquallstm/rnn/Shapelstm/rnn/stack*
T0*
_output_shapes
:
X
lstm/rnn/ConstConst*
valueB: *
dtype0*
_output_shapes
:
h
lstm/rnn/AllAlllstm/rnn/Equallstm/rnn/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
�
lstm/rnn/Assert/ConstConst*I
value@B> B8Expected shape for Tensor lstm/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
h
lstm/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
�
lstm/rnn/Assert/Assert/data_0Const*I
value@B> B8Expected shape for Tensor lstm/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
n
lstm/rnn/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
�
lstm/rnn/Assert/AssertAssertlstm/rnn/Alllstm/rnn/Assert/Assert/data_0lstm/rnn/stacklstm/rnn/Assert/Assert/data_2lstm/rnn/Shape*
T
2*
	summarize
x
lstm/rnn/CheckSeqLenIdentitylstm/rnn/sequence_length^lstm/rnn/Assert/Assert*
T0*
_output_shapes
:
e
lstm/rnn/Shape_1Const*!
valueB"}         *
dtype0*
_output_shapes
:
f
lstm/rnn/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
h
lstm/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
lstm/rnn/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
lstm/rnn/strided_sliceStridedSlicelstm/rnn/Shape_1lstm/rnn/strided_slice/stacklstm/rnn/strided_slice/stack_1lstm/rnn/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
Z
lstm/rnn/Const_1Const*
dtype0*
_output_shapes
:*
valueB:
[
lstm/rnn/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:
X
lstm/rnn/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
lstm/rnn/concat_1ConcatV2lstm/rnn/Const_1lstm/rnn/Const_2lstm/rnn/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
Y
lstm/rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
{
lstm/rnn/zerosFilllstm/rnn/concat_1lstm/rnn/zeros/Const*
T0*

index_type0*
_output_shapes
:	�
Z
lstm/rnn/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
y
lstm/rnn/MinMinlstm/rnn/CheckSeqLenlstm/rnn/Const_3*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Z
lstm/rnn/Const_4Const*
valueB: *
dtype0*
_output_shapes
:
y
lstm/rnn/MaxMaxlstm/rnn/CheckSeqLenlstm/rnn/Const_4*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
O
lstm/rnn/timeConst*
dtype0*
_output_shapes
: *
value	B : 
�
lstm/rnn/TensorArrayTensorArrayV3lstm/rnn/strided_slice*4
tensor_array_namelstm/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *
element_shape:	�*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
�
lstm/rnn/TensorArray_1TensorArrayV3lstm/rnn/strided_slice*
element_shape:	�*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*3
tensor_array_namelstm/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: 
v
!lstm/rnn/TensorArrayUnstack/ShapeConst*!
valueB"}         *
dtype0*
_output_shapes
:
y
/lstm/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
{
1lstm/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
{
1lstm/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
)lstm/rnn/TensorArrayUnstack/strided_sliceStridedSlice!lstm/rnn/TensorArrayUnstack/Shape/lstm/rnn/TensorArrayUnstack/strided_slice/stack1lstm/rnn/TensorArrayUnstack/strided_slice/stack_11lstm/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
i
'lstm/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
i
'lstm/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
!lstm/rnn/TensorArrayUnstack/rangeRange'lstm/rnn/TensorArrayUnstack/range/start)lstm/rnn/TensorArrayUnstack/strided_slice'lstm/rnn/TensorArrayUnstack/range/delta*

Tidx0*
_output_shapes
:}
�
Clstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3lstm/rnn/TensorArray_1!lstm/rnn/TensorArrayUnstack/rangelstm/rnn/transposelstm/rnn/TensorArray_1:1*
_output_shapes
: *
T0*%
_class
loc:@lstm/rnn/transpose
T
lstm/rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
^
lstm/rnn/MaximumMaximumlstm/rnn/Maximum/xlstm/rnn/Max*
T0*
_output_shapes
: 
f
lstm/rnn/MinimumMinimumlstm/rnn/strided_slicelstm/rnn/Maximum*
T0*
_output_shapes
: 
b
 lstm/rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
�
lstm/rnn/while/EnterEnter lstm/rnn/while/iteration_counter*
parallel_iterations *
_output_shapes
: *,

frame_namelstm/rnn/while/while_context*
T0*
is_constant( 
�
lstm/rnn/while/Enter_1Enterlstm/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *,

frame_namelstm/rnn/while/while_context
�
lstm/rnn/while/Enter_2Enterlstm/rnn/TensorArray:1*
parallel_iterations *
_output_shapes
: *,

frame_namelstm/rnn/while/while_context*
T0*
is_constant( 
�
lstm/rnn/while/Enter_3Enter2lstm/MultiRNNCellZeroState/LSTMCellZeroState/zeros*
parallel_iterations *
_output_shapes
:	�*,

frame_namelstm/rnn/while/while_context*
T0*
is_constant( 
�
lstm/rnn/while/Enter_4Enter4lstm/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	�*,

frame_namelstm/rnn/while/while_context
�
lstm/rnn/while/Enter_5Enter4lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros*
parallel_iterations *
_output_shapes
:	�*,

frame_namelstm/rnn/while/while_context*
T0*
is_constant( 
�
lstm/rnn/while/Enter_6Enter6lstm/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
:	�*,

frame_namelstm/rnn/while/while_context
}
lstm/rnn/while/MergeMergelstm/rnn/while/Enterlstm/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
�
lstm/rnn/while/Merge_1Mergelstm/rnn/while/Enter_1lstm/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
�
lstm/rnn/while/Merge_2Mergelstm/rnn/while/Enter_2lstm/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
�
lstm/rnn/while/Merge_3Mergelstm/rnn/while/Enter_3lstm/rnn/while/NextIteration_3*
T0*
N*!
_output_shapes
:	�: 
�
lstm/rnn/while/Merge_4Mergelstm/rnn/while/Enter_4lstm/rnn/while/NextIteration_4*
T0*
N*!
_output_shapes
:	�: 
�
lstm/rnn/while/Merge_5Mergelstm/rnn/while/Enter_5lstm/rnn/while/NextIteration_5*
N*!
_output_shapes
:	�: *
T0
�
lstm/rnn/while/Merge_6Mergelstm/rnn/while/Enter_6lstm/rnn/while/NextIteration_6*
T0*
N*!
_output_shapes
:	�: 
m
lstm/rnn/while/LessLesslstm/rnn/while/Mergelstm/rnn/while/Less/Enter*
T0*
_output_shapes
: 
�
lstm/rnn/while/Less/EnterEnterlstm/rnn/strided_slice*
parallel_iterations *
_output_shapes
: *,

frame_namelstm/rnn/while/while_context*
T0*
is_constant(
s
lstm/rnn/while/Less_1Lesslstm/rnn/while/Merge_1lstm/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
�
lstm/rnn/while/Less_1/EnterEnterlstm/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *,

frame_namelstm/rnn/while/while_context
k
lstm/rnn/while/LogicalAnd
LogicalAndlstm/rnn/while/Lesslstm/rnn/while/Less_1*
_output_shapes
: 
V
lstm/rnn/while/LoopCondLoopCondlstm/rnn/while/LogicalAnd*
_output_shapes
: 
�
lstm/rnn/while/SwitchSwitchlstm/rnn/while/Mergelstm/rnn/while/LoopCond*
T0*'
_class
loc:@lstm/rnn/while/Merge*
_output_shapes
: : 
�
lstm/rnn/while/Switch_1Switchlstm/rnn/while/Merge_1lstm/rnn/while/LoopCond*
T0*)
_class
loc:@lstm/rnn/while/Merge_1*
_output_shapes
: : 
�
lstm/rnn/while/Switch_2Switchlstm/rnn/while/Merge_2lstm/rnn/while/LoopCond*
T0*)
_class
loc:@lstm/rnn/while/Merge_2*
_output_shapes
: : 
�
lstm/rnn/while/Switch_3Switchlstm/rnn/while/Merge_3lstm/rnn/while/LoopCond*
T0*)
_class
loc:@lstm/rnn/while/Merge_3**
_output_shapes
:	�:	�
�
lstm/rnn/while/Switch_4Switchlstm/rnn/while/Merge_4lstm/rnn/while/LoopCond*
T0*)
_class
loc:@lstm/rnn/while/Merge_4**
_output_shapes
:	�:	�
�
lstm/rnn/while/Switch_5Switchlstm/rnn/while/Merge_5lstm/rnn/while/LoopCond*
T0*)
_class
loc:@lstm/rnn/while/Merge_5**
_output_shapes
:	�:	�
�
lstm/rnn/while/Switch_6Switchlstm/rnn/while/Merge_6lstm/rnn/while/LoopCond**
_output_shapes
:	�:	�*
T0*)
_class
loc:@lstm/rnn/while/Merge_6
]
lstm/rnn/while/IdentityIdentitylstm/rnn/while/Switch:1*
T0*
_output_shapes
: 
a
lstm/rnn/while/Identity_1Identitylstm/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
a
lstm/rnn/while/Identity_2Identitylstm/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
j
lstm/rnn/while/Identity_3Identitylstm/rnn/while/Switch_3:1*
T0*
_output_shapes
:	�
j
lstm/rnn/while/Identity_4Identitylstm/rnn/while/Switch_4:1*
T0*
_output_shapes
:	�
j
lstm/rnn/while/Identity_5Identitylstm/rnn/while/Switch_5:1*
T0*
_output_shapes
:	�
j
lstm/rnn/while/Identity_6Identitylstm/rnn/while/Switch_6:1*
T0*
_output_shapes
:	�
p
lstm/rnn/while/add/yConst^lstm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
i
lstm/rnn/while/addAddlstm/rnn/while/Identitylstm/rnn/while/add/y*
T0*
_output_shapes
: 
�
 lstm/rnn/while/TensorArrayReadV3TensorArrayReadV3&lstm/rnn/while/TensorArrayReadV3/Enterlstm/rnn/while/Identity_1(lstm/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
:	�
�
&lstm/rnn/while/TensorArrayReadV3/EnterEnterlstm/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namelstm/rnn/while/while_context
�
(lstm/rnn/while/TensorArrayReadV3/Enter_1EnterClstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *,

frame_namelstm/rnn/while/while_context
�
lstm/rnn/while/GreaterEqualGreaterEquallstm/rnn/while/Identity_1!lstm/rnn/while/GreaterEqual/Enter*
T0*
_output_shapes
:
�
!lstm/rnn/while/GreaterEqual/EnterEnterlstm/rnn/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*,

frame_namelstm/rnn/while/while_context
�
Plstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"�     *B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
dtype0*
_output_shapes
:
�
Nlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *���*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
dtype0*
_output_shapes
: 
�
Nlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
dtype0*
_output_shapes
: 
�
Xlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformPlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed *
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
seed2 
�
Nlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/subSubNlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxNlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
�
Nlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulMulXlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformNlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel* 
_output_shapes
:
��
�
Jlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniformAddNlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulNlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
�
/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
	container *
shape:
��
�
6lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AssignAssign/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelJlstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform*
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
4lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/readIdentity/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
T0* 
_output_shapes
:
��
�
?lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zerosConst*
valueB�*    *@
_class6
42loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
dtype0*
_output_shapes	
:�
�
-lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
VariableV2*
shared_name *@
_class6
42loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
4lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AssignAssign-lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias?lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes	
:�
�
2lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/readIdentity-lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes	
:�*
T0
�
>lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axisConst^lstm/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
9lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concatConcatV2 lstm/rnn/while/TensorArrayReadV3lstm/rnn/while/Identity_4>lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axis*
T0*
N*
_output_shapes
:	�*

Tidx0
�
9lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMulMatMul9lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat?lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter*
_output_shapes
:	�*
transpose_a( *
transpose_b( *
T0
�
?lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/EnterEnter4lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
��*,

frame_namelstm/rnn/while/while_context
�
:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAddBiasAdd9lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul@lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes
:	�
�
@lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/EnterEnter2lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:�*,

frame_namelstm/rnn/while/while_context
�
8lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/ConstConst^lstm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Blstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dimConst^lstm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
8lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/splitSplitBlstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dim:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd*
T0*@
_output_shapes.
,:	�:	�:	�:	�*
	num_split
�
8lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/yConst^lstm/rnn/while/Identity*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
6lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/addAdd:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:28lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/y*
_output_shapes
:	�*
T0
�
:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/SigmoidSigmoid6lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add*
_output_shapes
:	�*
T0
�
6lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mulMul:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoidlstm/rnn/while/Identity_3*
T0*
_output_shapes
:	�
�
<lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1Sigmoid8lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split*
T0*
_output_shapes
:	�
�
7lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/TanhTanh:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:1*
T0*
_output_shapes
:	�
�
8lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1Mul<lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_17lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh*
T0*
_output_shapes
:	�
�
8lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1Add6lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul8lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1*
T0*
_output_shapes
:	�
�
<lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2Sigmoid:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:3*
T0*
_output_shapes
:	�
�
9lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1Tanh8lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1*
T0*
_output_shapes
:	�
�
8lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2Mul<lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_29lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
_output_shapes
:	�*
T0
�
Plstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"      *B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
dtype0*
_output_shapes
:
�
Nlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *���*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
dtype0*
_output_shapes
: 
�
Nlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
dtype0*
_output_shapes
: 
�
Xlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformPlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/shape*
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
seed2 *
dtype0* 
_output_shapes
:
��*

seed 
�
Nlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/subSubNlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/maxNlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/min*
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
_output_shapes
: 
�
Nlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/mulMulXlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/RandomUniformNlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
�
Jlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniformAddNlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/mulNlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/min*
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel* 
_output_shapes
:
��
�
/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
	container 
�
6lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AssignAssign/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelJlstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
��
�
4lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/readIdentity/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
T0* 
_output_shapes
:
��
�
?lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*
valueB�*    *@
_class6
42loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias
�
-lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias
VariableV2*@
_class6
42loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
4lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AssignAssign-lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias?lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*@
_class6
42loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias
�
2lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/readIdentity-lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
T0*
_output_shapes	
:�
�
>lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat/axisConst^lstm/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
�
9lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concatConcatV28lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2lstm/rnn/while/Identity_6>lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat/axis*
T0*
N*
_output_shapes
:	�*

Tidx0
�
9lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMulMatMul9lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat?lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter*
_output_shapes
:	�*
transpose_a( *
transpose_b( *
T0
�
?lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/EnterEnter4lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
��*,

frame_namelstm/rnn/while/while_context
�
:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAddBiasAdd9lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul@lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes
:	�
�
@lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/EnterEnter2lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:�*,

frame_namelstm/rnn/while/while_context
�
8lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/ConstConst^lstm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
Blstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split/split_dimConst^lstm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
�
8lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/splitSplitBlstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split/split_dim:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd*
T0*@
_output_shapes.
,:	�:	�:	�:	�*
	num_split
�
8lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add/yConst^lstm/rnn/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
6lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/addAdd:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:28lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add/y*
_output_shapes
:	�*
T0
�
:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/SigmoidSigmoid6lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add*
T0*
_output_shapes
:	�
�
6lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mulMul:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoidlstm/rnn/while/Identity_5*
_output_shapes
:	�*
T0
�
<lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1Sigmoid8lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split*
T0*
_output_shapes
:	�
�
7lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/TanhTanh:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:1*
T0*
_output_shapes
:	�
�
8lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1Mul<lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_17lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0*
_output_shapes
:	�
�
8lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1Add6lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul8lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1*
_output_shapes
:	�*
T0
�
<lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2Sigmoid:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:3*
T0*
_output_shapes
:	�
�
9lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1Tanh8lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1*
_output_shapes
:	�*
T0
�
8lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2Mul<lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_29lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
T0*
_output_shapes
:	�
�
lstm/rnn/while/SelectSelectlstm/rnn/while/GreaterEquallstm/rnn/while/Select/Enter8lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0*K
_classA
?=loc:@lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*
_output_shapes
:	�
�
lstm/rnn/while/Select/EnterEnterlstm/rnn/zeros*
T0*K
_classA
?=loc:@lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:	�*,

frame_namelstm/rnn/while/while_context
�
lstm/rnn/while/Select_1Selectlstm/rnn/while/GreaterEquallstm/rnn/while/Identity_38lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1*
T0*K
_classA
?=loc:@lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1*
_output_shapes
:	�
�
lstm/rnn/while/Select_2Selectlstm/rnn/while/GreaterEquallstm/rnn/while/Identity_48lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0*K
_classA
?=loc:@lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:	�
�
lstm/rnn/while/Select_3Selectlstm/rnn/while/GreaterEquallstm/rnn/while/Identity_58lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1*
_output_shapes
:	�*
T0*K
_classA
?=loc:@lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1
�
lstm/rnn/while/Select_4Selectlstm/rnn/while/GreaterEquallstm/rnn/while/Identity_68lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0*K
_classA
?=loc:@lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*
_output_shapes
:	�
�
2lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV38lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterlstm/rnn/while/Identity_1lstm/rnn/while/Selectlstm/rnn/while/Identity_2*
_output_shapes
: *
T0*K
_classA
?=loc:@lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2
�
8lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm/rnn/TensorArray*
is_constant(*
_output_shapes
:*,

frame_namelstm/rnn/while/while_context*
T0*K
_classA
?=loc:@lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*
parallel_iterations 
r
lstm/rnn/while/add_1/yConst^lstm/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
lstm/rnn/while/add_1Addlstm/rnn/while/Identity_1lstm/rnn/while/add_1/y*
_output_shapes
: *
T0
b
lstm/rnn/while/NextIterationNextIterationlstm/rnn/while/add*
T0*
_output_shapes
: 
f
lstm/rnn/while/NextIteration_1NextIterationlstm/rnn/while/add_1*
_output_shapes
: *
T0
�
lstm/rnn/while/NextIteration_2NextIteration2lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
r
lstm/rnn/while/NextIteration_3NextIterationlstm/rnn/while/Select_1*
_output_shapes
:	�*
T0
r
lstm/rnn/while/NextIteration_4NextIterationlstm/rnn/while/Select_2*
T0*
_output_shapes
:	�
r
lstm/rnn/while/NextIteration_5NextIterationlstm/rnn/while/Select_3*
T0*
_output_shapes
:	�
r
lstm/rnn/while/NextIteration_6NextIterationlstm/rnn/while/Select_4*
T0*
_output_shapes
:	�
S
lstm/rnn/while/ExitExitlstm/rnn/while/Switch*
T0*
_output_shapes
: 
W
lstm/rnn/while/Exit_1Exitlstm/rnn/while/Switch_1*
T0*
_output_shapes
: 
W
lstm/rnn/while/Exit_2Exitlstm/rnn/while/Switch_2*
T0*
_output_shapes
: 
`
lstm/rnn/while/Exit_3Exitlstm/rnn/while/Switch_3*
T0*
_output_shapes
:	�
`
lstm/rnn/while/Exit_4Exitlstm/rnn/while/Switch_4*
_output_shapes
:	�*
T0
`
lstm/rnn/while/Exit_5Exitlstm/rnn/while/Switch_5*
T0*
_output_shapes
:	�
`
lstm/rnn/while/Exit_6Exitlstm/rnn/while/Switch_6*
T0*
_output_shapes
:	�
�
+lstm/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lstm/rnn/TensorArraylstm/rnn/while/Exit_2*'
_class
loc:@lstm/rnn/TensorArray*
_output_shapes
: 
�
%lstm/rnn/TensorArrayStack/range/startConst*
value	B : *'
_class
loc:@lstm/rnn/TensorArray*
dtype0*
_output_shapes
: 
�
%lstm/rnn/TensorArrayStack/range/deltaConst*
value	B :*'
_class
loc:@lstm/rnn/TensorArray*
dtype0*
_output_shapes
: 
�
lstm/rnn/TensorArrayStack/rangeRange%lstm/rnn/TensorArrayStack/range/start+lstm/rnn/TensorArrayStack/TensorArraySizeV3%lstm/rnn/TensorArrayStack/range/delta*'
_class
loc:@lstm/rnn/TensorArray*#
_output_shapes
:���������*

Tidx0
�
-lstm/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm/rnn/TensorArraylstm/rnn/TensorArrayStack/rangelstm/rnn/while/Exit_2*'
_class
loc:@lstm/rnn/TensorArray*
dtype0*#
_output_shapes
:}�*
element_shape:	�
a
lstm/rnn/Const_5Const*
valueB"}      *
dtype0*
_output_shapes
:
[
lstm/rnn/Const_6Const*
valueB:�*
dtype0*
_output_shapes
:
Q
lstm/rnn/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
X
lstm/rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
X
lstm/rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
�
lstm/rnn/range_1Rangelstm/rnn/range_1/startlstm/rnn/Rank_1lstm/rnn/range_1/delta*
_output_shapes
:*

Tidx0
k
lstm/rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
X
lstm/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
lstm/rnn/concat_2ConcatV2lstm/rnn/concat_2/values_0lstm/rnn/range_1lstm/rnn/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
lstm/rnn/transpose_1	Transpose-lstm/rnn/TensorArrayStack/TensorArrayGatherV3lstm/rnn/concat_2*#
_output_shapes
:}�*
Tperm0*
T0
c
lstm/Reshape/shapeConst*
valueB"�����   *
dtype0*
_output_shapes
:
y
lstm/ReshapeReshapelstm/rnn/transpose_1lstm/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	}�
�
.lstm/weight/Initializer/truncated_normal/shapeConst*
valueB"�      *
_class
loc:@lstm/weight*
dtype0*
_output_shapes
:
�
-lstm/weight/Initializer/truncated_normal/meanConst*
valueB
 *    *
_class
loc:@lstm/weight*
dtype0*
_output_shapes
: 
�
/lstm/weight/Initializer/truncated_normal/stddevConst*
valueB
 *���=*
_class
loc:@lstm/weight*
dtype0*
_output_shapes
: 
�
8lstm/weight/Initializer/truncated_normal/TruncatedNormalTruncatedNormal.lstm/weight/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
:	�*

seed *
T0*
_class
loc:@lstm/weight*
seed2 
�
,lstm/weight/Initializer/truncated_normal/mulMul8lstm/weight/Initializer/truncated_normal/TruncatedNormal/lstm/weight/Initializer/truncated_normal/stddev*
T0*
_class
loc:@lstm/weight*
_output_shapes
:	�
�
(lstm/weight/Initializer/truncated_normalAdd,lstm/weight/Initializer/truncated_normal/mul-lstm/weight/Initializer/truncated_normal/mean*
T0*
_class
loc:@lstm/weight*
_output_shapes
:	�
�
lstm/weight
VariableV2*
dtype0*
_output_shapes
:	�*
shared_name *
_class
loc:@lstm/weight*
	container *
shape:	�
�
lstm/weight/AssignAssignlstm/weight(lstm/weight/Initializer/truncated_normal*
T0*
_class
loc:@lstm/weight*
validate_shape(*
_output_shapes
:	�*
use_locking(
s
lstm/weight/readIdentitylstm/weight*
T0*
_class
loc:@lstm/weight*
_output_shapes
:	�
�
lstm/bias/Initializer/ConstConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@lstm/bias
�
	lstm/bias
VariableV2*
shared_name *
_class
loc:@lstm/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
lstm/bias/AssignAssign	lstm/biaslstm/bias/Initializer/Const*
T0*
_class
loc:@lstm/bias*
validate_shape(*
_output_shapes
:*
use_locking(
h
lstm/bias/readIdentity	lstm/bias*
T0*
_class
loc:@lstm/bias*
_output_shapes
:
�
lstm/MatMulMatMullstm/Reshapelstm/weight/read*
T0*
_output_shapes

:}*
transpose_a( *
transpose_b( 
U
lstm/addAddlstm/MatMullstm/bias/read*
_output_shapes

:}*
T0
i
lstm/Reshape_1/shapeConst*!
valueB"   ����   *
dtype0*
_output_shapes
:
t
lstm/Reshape_1Reshapelstm/addlstm/Reshape_1/shape*
T0*
Tshape0*"
_output_shapes
:}
e
transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          
t
transpose_1	Transposelstm/Reshape_1transpose_1/perm*
T0*"
_output_shapes
:}*
Tperm0
�
CTCBeamSearchDecoderCTCBeamSearchDecodertranspose_1Fill*
	top_paths*F
_output_shapes4
2:���������:���������::*

beam_widthd*
merge_repeated( 
f
SparseToDense/default_valueConst*
valueB	 R
���������*
dtype0	*
_output_shapes
: 
�
SparseToDenseSparseToDenseCTCBeamSearchDecoderCTCBeamSearchDecoder:2CTCBeamSearchDecoder:1SparseToDense/default_value*0
_output_shapes
:������������������*
Tindices0	*
validate_indices(*
T0	
W
yIdentitySparseToDense*
T0	*0
_output_shapes
:������������������
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
�
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:$*�
value�B�$B*cnn/layer-conv1-1/batch-normalization/betaB+cnn/layer-conv1-1/batch-normalization/gammaB1cnn/layer-conv1-1/batch-normalization/moving_meanB5cnn/layer-conv1-1/batch-normalization/moving_varianceBcnn/layer-conv1-1/biasBcnn/layer-conv1-1/weightB(cnn/layer-conv1/batch-normalization/betaB)cnn/layer-conv1/batch-normalization/gammaB/cnn/layer-conv1/batch-normalization/moving_meanB3cnn/layer-conv1/batch-normalization/moving_varianceBcnn/layer-conv1/biasBcnn/layer-conv1/weightB(cnn/layer-conv2/batch-normalization/betaB)cnn/layer-conv2/batch-normalization/gammaB/cnn/layer-conv2/batch-normalization/moving_meanB3cnn/layer-conv2/batch-normalization/moving_varianceBcnn/layer-conv2/biasBcnn/layer-conv2/weightB*cnn/layer-conv3-1/batch-normalization/betaB+cnn/layer-conv3-1/batch-normalization/gammaB1cnn/layer-conv3-1/batch-normalization/moving_meanB5cnn/layer-conv3-1/batch-normalization/moving_varianceBcnn/layer-conv3-1/biasBcnn/layer-conv3-1/weightB(cnn/layer-conv3/batch-normalization/betaB)cnn/layer-conv3/batch-normalization/gammaB/cnn/layer-conv3/batch-normalization/moving_meanB3cnn/layer-conv3/batch-normalization/moving_varianceBcnn/layer-conv3/biasBcnn/layer-conv3/weightB	lstm/biasB-lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/biasB/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB-lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/biasB/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelBlstm/weight
�
save/SaveV2/shape_and_slicesConst*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:$
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices*cnn/layer-conv1-1/batch-normalization/beta+cnn/layer-conv1-1/batch-normalization/gamma1cnn/layer-conv1-1/batch-normalization/moving_mean5cnn/layer-conv1-1/batch-normalization/moving_variancecnn/layer-conv1-1/biascnn/layer-conv1-1/weight(cnn/layer-conv1/batch-normalization/beta)cnn/layer-conv1/batch-normalization/gamma/cnn/layer-conv1/batch-normalization/moving_mean3cnn/layer-conv1/batch-normalization/moving_variancecnn/layer-conv1/biascnn/layer-conv1/weight(cnn/layer-conv2/batch-normalization/beta)cnn/layer-conv2/batch-normalization/gamma/cnn/layer-conv2/batch-normalization/moving_mean3cnn/layer-conv2/batch-normalization/moving_variancecnn/layer-conv2/biascnn/layer-conv2/weight*cnn/layer-conv3-1/batch-normalization/beta+cnn/layer-conv3-1/batch-normalization/gamma1cnn/layer-conv3-1/batch-normalization/moving_mean5cnn/layer-conv3-1/batch-normalization/moving_variancecnn/layer-conv3-1/biascnn/layer-conv3-1/weight(cnn/layer-conv3/batch-normalization/beta)cnn/layer-conv3/batch-normalization/gamma/cnn/layer-conv3/batch-normalization/moving_mean3cnn/layer-conv3/batch-normalization/moving_variancecnn/layer-conv3/biascnn/layer-conv3/weight	lstm/bias-lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel-lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernellstm/weight*2
dtypes(
&2$
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�$B*cnn/layer-conv1-1/batch-normalization/betaB+cnn/layer-conv1-1/batch-normalization/gammaB1cnn/layer-conv1-1/batch-normalization/moving_meanB5cnn/layer-conv1-1/batch-normalization/moving_varianceBcnn/layer-conv1-1/biasBcnn/layer-conv1-1/weightB(cnn/layer-conv1/batch-normalization/betaB)cnn/layer-conv1/batch-normalization/gammaB/cnn/layer-conv1/batch-normalization/moving_meanB3cnn/layer-conv1/batch-normalization/moving_varianceBcnn/layer-conv1/biasBcnn/layer-conv1/weightB(cnn/layer-conv2/batch-normalization/betaB)cnn/layer-conv2/batch-normalization/gammaB/cnn/layer-conv2/batch-normalization/moving_meanB3cnn/layer-conv2/batch-normalization/moving_varianceBcnn/layer-conv2/biasBcnn/layer-conv2/weightB*cnn/layer-conv3-1/batch-normalization/betaB+cnn/layer-conv3-1/batch-normalization/gammaB1cnn/layer-conv3-1/batch-normalization/moving_meanB5cnn/layer-conv3-1/batch-normalization/moving_varianceBcnn/layer-conv3-1/biasBcnn/layer-conv3-1/weightB(cnn/layer-conv3/batch-normalization/betaB)cnn/layer-conv3/batch-normalization/gammaB/cnn/layer-conv3/batch-normalization/moving_meanB3cnn/layer-conv3/batch-normalization/moving_varianceBcnn/layer-conv3/biasBcnn/layer-conv3/weightB	lstm/biasB-lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/biasB/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB-lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/biasB/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelBlstm/weight*
dtype0*
_output_shapes
:$
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:$*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*2
dtypes(
&2$*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::
�
save/AssignAssign*cnn/layer-conv1-1/batch-normalization/betasave/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*=
_class3
1/loc:@cnn/layer-conv1-1/batch-normalization/beta
�
save/Assign_1Assign+cnn/layer-conv1-1/batch-normalization/gammasave/RestoreV2:1*
use_locking(*
T0*>
_class4
20loc:@cnn/layer-conv1-1/batch-normalization/gamma*
validate_shape(*
_output_shapes
: 
�
save/Assign_2Assign1cnn/layer-conv1-1/batch-normalization/moving_meansave/RestoreV2:2*
use_locking(*
T0*D
_class:
86loc:@cnn/layer-conv1-1/batch-normalization/moving_mean*
validate_shape(*
_output_shapes
: 
�
save/Assign_3Assign5cnn/layer-conv1-1/batch-normalization/moving_variancesave/RestoreV2:3*
T0*H
_class>
<:loc:@cnn/layer-conv1-1/batch-normalization/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_4Assigncnn/layer-conv1-1/biassave/RestoreV2:4*
T0*)
_class
loc:@cnn/layer-conv1-1/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_5Assigncnn/layer-conv1-1/weightsave/RestoreV2:5*
use_locking(*
T0*+
_class!
loc:@cnn/layer-conv1-1/weight*
validate_shape(*&
_output_shapes
:  
�
save/Assign_6Assign(cnn/layer-conv1/batch-normalization/betasave/RestoreV2:6*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*;
_class1
/-loc:@cnn/layer-conv1/batch-normalization/beta
�
save/Assign_7Assign)cnn/layer-conv1/batch-normalization/gammasave/RestoreV2:7*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*<
_class2
0.loc:@cnn/layer-conv1/batch-normalization/gamma
�
save/Assign_8Assign/cnn/layer-conv1/batch-normalization/moving_meansave/RestoreV2:8*
use_locking(*
T0*B
_class8
64loc:@cnn/layer-conv1/batch-normalization/moving_mean*
validate_shape(*
_output_shapes
: 
�
save/Assign_9Assign3cnn/layer-conv1/batch-normalization/moving_variancesave/RestoreV2:9*
use_locking(*
T0*F
_class<
:8loc:@cnn/layer-conv1/batch-normalization/moving_variance*
validate_shape(*
_output_shapes
: 
�
save/Assign_10Assigncnn/layer-conv1/biassave/RestoreV2:10*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@cnn/layer-conv1/bias
�
save/Assign_11Assigncnn/layer-conv1/weightsave/RestoreV2:11*
use_locking(*
T0*)
_class
loc:@cnn/layer-conv1/weight*
validate_shape(*&
_output_shapes
: 
�
save/Assign_12Assign(cnn/layer-conv2/batch-normalization/betasave/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*;
_class1
/-loc:@cnn/layer-conv2/batch-normalization/beta
�
save/Assign_13Assign)cnn/layer-conv2/batch-normalization/gammasave/RestoreV2:13*
use_locking(*
T0*<
_class2
0.loc:@cnn/layer-conv2/batch-normalization/gamma*
validate_shape(*
_output_shapes
:@
�
save/Assign_14Assign/cnn/layer-conv2/batch-normalization/moving_meansave/RestoreV2:14*
use_locking(*
T0*B
_class8
64loc:@cnn/layer-conv2/batch-normalization/moving_mean*
validate_shape(*
_output_shapes
:@
�
save/Assign_15Assign3cnn/layer-conv2/batch-normalization/moving_variancesave/RestoreV2:15*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*F
_class<
:8loc:@cnn/layer-conv2/batch-normalization/moving_variance
�
save/Assign_16Assigncnn/layer-conv2/biassave/RestoreV2:16*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@cnn/layer-conv2/bias
�
save/Assign_17Assigncnn/layer-conv2/weightsave/RestoreV2:17*
use_locking(*
T0*)
_class
loc:@cnn/layer-conv2/weight*
validate_shape(*&
_output_shapes
: @
�
save/Assign_18Assign*cnn/layer-conv3-1/batch-normalization/betasave/RestoreV2:18*
use_locking(*
T0*=
_class3
1/loc:@cnn/layer-conv3-1/batch-normalization/beta*
validate_shape(*
_output_shapes
:@
�
save/Assign_19Assign+cnn/layer-conv3-1/batch-normalization/gammasave/RestoreV2:19*
T0*>
_class4
20loc:@cnn/layer-conv3-1/batch-normalization/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save/Assign_20Assign1cnn/layer-conv3-1/batch-normalization/moving_meansave/RestoreV2:20*
T0*D
_class:
86loc:@cnn/layer-conv3-1/batch-normalization/moving_mean*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save/Assign_21Assign5cnn/layer-conv3-1/batch-normalization/moving_variancesave/RestoreV2:21*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*H
_class>
<:loc:@cnn/layer-conv3-1/batch-normalization/moving_variance
�
save/Assign_22Assigncnn/layer-conv3-1/biassave/RestoreV2:22*
use_locking(*
T0*)
_class
loc:@cnn/layer-conv3-1/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_23Assigncnn/layer-conv3-1/weightsave/RestoreV2:23*
T0*+
_class!
loc:@cnn/layer-conv3-1/weight*
validate_shape(*&
_output_shapes
:@@*
use_locking(
�
save/Assign_24Assign(cnn/layer-conv3/batch-normalization/betasave/RestoreV2:24*
use_locking(*
T0*;
_class1
/-loc:@cnn/layer-conv3/batch-normalization/beta*
validate_shape(*
_output_shapes
:@
�
save/Assign_25Assign)cnn/layer-conv3/batch-normalization/gammasave/RestoreV2:25*
use_locking(*
T0*<
_class2
0.loc:@cnn/layer-conv3/batch-normalization/gamma*
validate_shape(*
_output_shapes
:@
�
save/Assign_26Assign/cnn/layer-conv3/batch-normalization/moving_meansave/RestoreV2:26*
T0*B
_class8
64loc:@cnn/layer-conv3/batch-normalization/moving_mean*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save/Assign_27Assign3cnn/layer-conv3/batch-normalization/moving_variancesave/RestoreV2:27*
use_locking(*
T0*F
_class<
:8loc:@cnn/layer-conv3/batch-normalization/moving_variance*
validate_shape(*
_output_shapes
:@
�
save/Assign_28Assigncnn/layer-conv3/biassave/RestoreV2:28*
use_locking(*
T0*'
_class
loc:@cnn/layer-conv3/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_29Assigncnn/layer-conv3/weightsave/RestoreV2:29*
T0*)
_class
loc:@cnn/layer-conv3/weight*
validate_shape(*&
_output_shapes
:@@*
use_locking(
�
save/Assign_30Assign	lstm/biassave/RestoreV2:30*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@lstm/bias
�
save/Assign_31Assign-lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/biassave/RestoreV2:31*
use_locking(*
T0*@
_class6
42loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes	
:�
�
save/Assign_32Assign/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave/RestoreV2:32*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
�
save/Assign_33Assign-lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/biassave/RestoreV2:33*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*@
_class6
42loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias
�
save/Assign_34Assign/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave/RestoreV2:34*
use_locking(*
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
��
�
save/Assign_35Assignlstm/weightsave/RestoreV2:35*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*
_class
loc:@lstm/weight
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp2^cnn/layer-conv1-1/batch-normalization/beta/Assign3^cnn/layer-conv1-1/batch-normalization/gamma/Assign9^cnn/layer-conv1-1/batch-normalization/moving_mean/Assign=^cnn/layer-conv1-1/batch-normalization/moving_variance/Assign^cnn/layer-conv1-1/bias/Assign ^cnn/layer-conv1-1/weight/Assign0^cnn/layer-conv1/batch-normalization/beta/Assign1^cnn/layer-conv1/batch-normalization/gamma/Assign7^cnn/layer-conv1/batch-normalization/moving_mean/Assign;^cnn/layer-conv1/batch-normalization/moving_variance/Assign^cnn/layer-conv1/bias/Assign^cnn/layer-conv1/weight/Assign0^cnn/layer-conv2/batch-normalization/beta/Assign1^cnn/layer-conv2/batch-normalization/gamma/Assign7^cnn/layer-conv2/batch-normalization/moving_mean/Assign;^cnn/layer-conv2/batch-normalization/moving_variance/Assign^cnn/layer-conv2/bias/Assign^cnn/layer-conv2/weight/Assign2^cnn/layer-conv3-1/batch-normalization/beta/Assign3^cnn/layer-conv3-1/batch-normalization/gamma/Assign9^cnn/layer-conv3-1/batch-normalization/moving_mean/Assign=^cnn/layer-conv3-1/batch-normalization/moving_variance/Assign^cnn/layer-conv3-1/bias/Assign ^cnn/layer-conv3-1/weight/Assign0^cnn/layer-conv3/batch-normalization/beta/Assign1^cnn/layer-conv3/batch-normalization/gamma/Assign7^cnn/layer-conv3/batch-normalization/moving_mean/Assign;^cnn/layer-conv3/batch-normalization/moving_variance/Assign^cnn/layer-conv3/bias/Assign^cnn/layer-conv3/weight/Assign^lstm/bias/Assign5^lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign7^lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign5^lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign7^lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign^lstm/weight/Assign
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_e56632f0fd1a49cab385e1d022fa8210/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
�
save_1/SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�$B*cnn/layer-conv1-1/batch-normalization/betaB+cnn/layer-conv1-1/batch-normalization/gammaB1cnn/layer-conv1-1/batch-normalization/moving_meanB5cnn/layer-conv1-1/batch-normalization/moving_varianceBcnn/layer-conv1-1/biasBcnn/layer-conv1-1/weightB(cnn/layer-conv1/batch-normalization/betaB)cnn/layer-conv1/batch-normalization/gammaB/cnn/layer-conv1/batch-normalization/moving_meanB3cnn/layer-conv1/batch-normalization/moving_varianceBcnn/layer-conv1/biasBcnn/layer-conv1/weightB(cnn/layer-conv2/batch-normalization/betaB)cnn/layer-conv2/batch-normalization/gammaB/cnn/layer-conv2/batch-normalization/moving_meanB3cnn/layer-conv2/batch-normalization/moving_varianceBcnn/layer-conv2/biasBcnn/layer-conv2/weightB*cnn/layer-conv3-1/batch-normalization/betaB+cnn/layer-conv3-1/batch-normalization/gammaB1cnn/layer-conv3-1/batch-normalization/moving_meanB5cnn/layer-conv3-1/batch-normalization/moving_varianceBcnn/layer-conv3-1/biasBcnn/layer-conv3-1/weightB(cnn/layer-conv3/batch-normalization/betaB)cnn/layer-conv3/batch-normalization/gammaB/cnn/layer-conv3/batch-normalization/moving_meanB3cnn/layer-conv3/batch-normalization/moving_varianceBcnn/layer-conv3/biasBcnn/layer-conv3/weightB	lstm/biasB-lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/biasB/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB-lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/biasB/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelBlstm/weight*
dtype0*
_output_shapes
:$
�
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:$
�
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices*cnn/layer-conv1-1/batch-normalization/beta+cnn/layer-conv1-1/batch-normalization/gamma1cnn/layer-conv1-1/batch-normalization/moving_mean5cnn/layer-conv1-1/batch-normalization/moving_variancecnn/layer-conv1-1/biascnn/layer-conv1-1/weight(cnn/layer-conv1/batch-normalization/beta)cnn/layer-conv1/batch-normalization/gamma/cnn/layer-conv1/batch-normalization/moving_mean3cnn/layer-conv1/batch-normalization/moving_variancecnn/layer-conv1/biascnn/layer-conv1/weight(cnn/layer-conv2/batch-normalization/beta)cnn/layer-conv2/batch-normalization/gamma/cnn/layer-conv2/batch-normalization/moving_mean3cnn/layer-conv2/batch-normalization/moving_variancecnn/layer-conv2/biascnn/layer-conv2/weight*cnn/layer-conv3-1/batch-normalization/beta+cnn/layer-conv3-1/batch-normalization/gamma1cnn/layer-conv3-1/batch-normalization/moving_mean5cnn/layer-conv3-1/batch-normalization/moving_variancecnn/layer-conv3-1/biascnn/layer-conv3-1/weight(cnn/layer-conv3/batch-normalization/beta)cnn/layer-conv3/batch-normalization/gamma/cnn/layer-conv3/batch-normalization/moving_mean3cnn/layer-conv3/batch-normalization/moving_variancecnn/layer-conv3/biascnn/layer-conv3/weight	lstm/bias-lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel-lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernellstm/weight"/device:CPU:0*2
dtypes(
&2$
�
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
�
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
�
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
�
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
�
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:$*�
value�B�$B*cnn/layer-conv1-1/batch-normalization/betaB+cnn/layer-conv1-1/batch-normalization/gammaB1cnn/layer-conv1-1/batch-normalization/moving_meanB5cnn/layer-conv1-1/batch-normalization/moving_varianceBcnn/layer-conv1-1/biasBcnn/layer-conv1-1/weightB(cnn/layer-conv1/batch-normalization/betaB)cnn/layer-conv1/batch-normalization/gammaB/cnn/layer-conv1/batch-normalization/moving_meanB3cnn/layer-conv1/batch-normalization/moving_varianceBcnn/layer-conv1/biasBcnn/layer-conv1/weightB(cnn/layer-conv2/batch-normalization/betaB)cnn/layer-conv2/batch-normalization/gammaB/cnn/layer-conv2/batch-normalization/moving_meanB3cnn/layer-conv2/batch-normalization/moving_varianceBcnn/layer-conv2/biasBcnn/layer-conv2/weightB*cnn/layer-conv3-1/batch-normalization/betaB+cnn/layer-conv3-1/batch-normalization/gammaB1cnn/layer-conv3-1/batch-normalization/moving_meanB5cnn/layer-conv3-1/batch-normalization/moving_varianceBcnn/layer-conv3-1/biasBcnn/layer-conv3-1/weightB(cnn/layer-conv3/batch-normalization/betaB)cnn/layer-conv3/batch-normalization/gammaB/cnn/layer-conv3/batch-normalization/moving_meanB3cnn/layer-conv3/batch-normalization/moving_varianceBcnn/layer-conv3/biasBcnn/layer-conv3/weightB	lstm/biasB-lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/biasB/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB-lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/biasB/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelBlstm/weight
�
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:$
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*2
dtypes(
&2$*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::
�
save_1/AssignAssign*cnn/layer-conv1-1/batch-normalization/betasave_1/RestoreV2*
use_locking(*
T0*=
_class3
1/loc:@cnn/layer-conv1-1/batch-normalization/beta*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_1Assign+cnn/layer-conv1-1/batch-normalization/gammasave_1/RestoreV2:1*
T0*>
_class4
20loc:@cnn/layer-conv1-1/batch-normalization/gamma*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_2Assign1cnn/layer-conv1-1/batch-normalization/moving_meansave_1/RestoreV2:2*
use_locking(*
T0*D
_class:
86loc:@cnn/layer-conv1-1/batch-normalization/moving_mean*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_3Assign5cnn/layer-conv1-1/batch-normalization/moving_variancesave_1/RestoreV2:3*
use_locking(*
T0*H
_class>
<:loc:@cnn/layer-conv1-1/batch-normalization/moving_variance*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_4Assigncnn/layer-conv1-1/biassave_1/RestoreV2:4*
use_locking(*
T0*)
_class
loc:@cnn/layer-conv1-1/bias*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_5Assigncnn/layer-conv1-1/weightsave_1/RestoreV2:5*
use_locking(*
T0*+
_class!
loc:@cnn/layer-conv1-1/weight*
validate_shape(*&
_output_shapes
:  
�
save_1/Assign_6Assign(cnn/layer-conv1/batch-normalization/betasave_1/RestoreV2:6*
T0*;
_class1
/-loc:@cnn/layer-conv1/batch-normalization/beta*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_7Assign)cnn/layer-conv1/batch-normalization/gammasave_1/RestoreV2:7*
use_locking(*
T0*<
_class2
0.loc:@cnn/layer-conv1/batch-normalization/gamma*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_8Assign/cnn/layer-conv1/batch-normalization/moving_meansave_1/RestoreV2:8*
use_locking(*
T0*B
_class8
64loc:@cnn/layer-conv1/batch-normalization/moving_mean*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_9Assign3cnn/layer-conv1/batch-normalization/moving_variancesave_1/RestoreV2:9*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*F
_class<
:8loc:@cnn/layer-conv1/batch-normalization/moving_variance
�
save_1/Assign_10Assigncnn/layer-conv1/biassave_1/RestoreV2:10*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@cnn/layer-conv1/bias
�
save_1/Assign_11Assigncnn/layer-conv1/weightsave_1/RestoreV2:11*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*)
_class
loc:@cnn/layer-conv1/weight
�
save_1/Assign_12Assign(cnn/layer-conv2/batch-normalization/betasave_1/RestoreV2:12*
use_locking(*
T0*;
_class1
/-loc:@cnn/layer-conv2/batch-normalization/beta*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_13Assign)cnn/layer-conv2/batch-normalization/gammasave_1/RestoreV2:13*
use_locking(*
T0*<
_class2
0.loc:@cnn/layer-conv2/batch-normalization/gamma*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_14Assign/cnn/layer-conv2/batch-normalization/moving_meansave_1/RestoreV2:14*
use_locking(*
T0*B
_class8
64loc:@cnn/layer-conv2/batch-normalization/moving_mean*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_15Assign3cnn/layer-conv2/batch-normalization/moving_variancesave_1/RestoreV2:15*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*F
_class<
:8loc:@cnn/layer-conv2/batch-normalization/moving_variance
�
save_1/Assign_16Assigncnn/layer-conv2/biassave_1/RestoreV2:16*
T0*'
_class
loc:@cnn/layer-conv2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_17Assigncnn/layer-conv2/weightsave_1/RestoreV2:17*
use_locking(*
T0*)
_class
loc:@cnn/layer-conv2/weight*
validate_shape(*&
_output_shapes
: @
�
save_1/Assign_18Assign*cnn/layer-conv3-1/batch-normalization/betasave_1/RestoreV2:18*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*=
_class3
1/loc:@cnn/layer-conv3-1/batch-normalization/beta
�
save_1/Assign_19Assign+cnn/layer-conv3-1/batch-normalization/gammasave_1/RestoreV2:19*
use_locking(*
T0*>
_class4
20loc:@cnn/layer-conv3-1/batch-normalization/gamma*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_20Assign1cnn/layer-conv3-1/batch-normalization/moving_meansave_1/RestoreV2:20*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*D
_class:
86loc:@cnn/layer-conv3-1/batch-normalization/moving_mean
�
save_1/Assign_21Assign5cnn/layer-conv3-1/batch-normalization/moving_variancesave_1/RestoreV2:21*
T0*H
_class>
<:loc:@cnn/layer-conv3-1/batch-normalization/moving_variance*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_22Assigncnn/layer-conv3-1/biassave_1/RestoreV2:22*
use_locking(*
T0*)
_class
loc:@cnn/layer-conv3-1/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_23Assigncnn/layer-conv3-1/weightsave_1/RestoreV2:23*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*+
_class!
loc:@cnn/layer-conv3-1/weight
�
save_1/Assign_24Assign(cnn/layer-conv3/batch-normalization/betasave_1/RestoreV2:24*
use_locking(*
T0*;
_class1
/-loc:@cnn/layer-conv3/batch-normalization/beta*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_25Assign)cnn/layer-conv3/batch-normalization/gammasave_1/RestoreV2:25*
use_locking(*
T0*<
_class2
0.loc:@cnn/layer-conv3/batch-normalization/gamma*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_26Assign/cnn/layer-conv3/batch-normalization/moving_meansave_1/RestoreV2:26*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*B
_class8
64loc:@cnn/layer-conv3/batch-normalization/moving_mean
�
save_1/Assign_27Assign3cnn/layer-conv3/batch-normalization/moving_variancesave_1/RestoreV2:27*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*F
_class<
:8loc:@cnn/layer-conv3/batch-normalization/moving_variance
�
save_1/Assign_28Assigncnn/layer-conv3/biassave_1/RestoreV2:28*
use_locking(*
T0*'
_class
loc:@cnn/layer-conv3/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_29Assigncnn/layer-conv3/weightsave_1/RestoreV2:29*
use_locking(*
T0*)
_class
loc:@cnn/layer-conv3/weight*
validate_shape(*&
_output_shapes
:@@
�
save_1/Assign_30Assign	lstm/biassave_1/RestoreV2:30*
T0*
_class
loc:@lstm/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_1/Assign_31Assign-lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_1/RestoreV2:31*
T0*@
_class6
42loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save_1/Assign_32Assign/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_1/RestoreV2:32*
use_locking(*
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
��
�
save_1/Assign_33Assign-lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_1/RestoreV2:33*
use_locking(*
T0*@
_class6
42loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
validate_shape(*
_output_shapes	
:�
�
save_1/Assign_34Assign/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_1/RestoreV2:34*
T0*B
_class8
64loc:@lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
save_1/Assign_35Assignlstm/weightsave_1/RestoreV2:35*
T0*
_class
loc:@lstm/weight*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"�Q
cond_context�Q�Q
�
 decode_image/cond_jpeg/cond_text decode_image/cond_jpeg/pred_id:0!decode_image/cond_jpeg/switch_t:0 *�
-decode_image/cond_jpeg/Assert/Assert/data_0:0
%decode_image/cond_jpeg/Assert/Const:0
*decode_image/cond_jpeg/DecodeJpeg/Switch:1
#decode_image/cond_jpeg/DecodeJpeg:0
!decode_image/cond_jpeg/Identity:0
.decode_image/cond_jpeg/check_jpeg_channels/x:0
.decode_image/cond_jpeg/check_jpeg_channels/y:0
,decode_image/cond_jpeg/check_jpeg_channels:0
 decode_image/cond_jpeg/pred_id:0
!decode_image/cond_jpeg/switch_t:0
x:01
x:0*decode_image/cond_jpeg/DecodeJpeg/Switch:1D
 decode_image/cond_jpeg/pred_id:0 decode_image/cond_jpeg/pred_id:0
�L
"decode_image/cond_jpeg/cond_text_1 decode_image/cond_jpeg/pred_id:0!decode_image/cond_jpeg/switch_f:0*�
decode_image/Substr:0
2decode_image/cond_jpeg/cond_png/DecodePng/Switch:1
+decode_image/cond_jpeg/cond_png/DecodePng:0
*decode_image/cond_jpeg/cond_png/Identity:0
'decode_image/cond_jpeg/cond_png/Merge:0
'decode_image/cond_jpeg/cond_png/Merge:1
(decode_image/cond_jpeg/cond_png/Switch:0
(decode_image/cond_jpeg/cond_png/Switch:1
?decode_image/cond_jpeg/cond_png/cond_gif/Assert/Assert/data_0:0
7decode_image/cond_jpeg/cond_png/cond_gif/Assert/Const:0
Adecode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Assert/data_0:0
9decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Const:0
Adecode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Assert/data_0:0
9decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Const:0
4decode_image/cond_jpeg/cond_png/cond_gif/DecodeBmp:0
;decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch:0
=decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch_1:1
4decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif:0
3decode_image/cond_jpeg/cond_png/cond_gif/Identity:0
5decode_image/cond_jpeg/cond_png/cond_gif/Identity_1:0
5decode_image/cond_jpeg/cond_png/cond_gif/LogicalAnd:0
0decode_image/cond_jpeg/cond_png/cond_gif/Merge:0
0decode_image/cond_jpeg/cond_png/cond_gif/Merge:1
8decode_image/cond_jpeg/cond_png/cond_gif/Substr/Switch:0
5decode_image/cond_jpeg/cond_png/cond_gif/Substr/len:0
5decode_image/cond_jpeg/cond_png/cond_gif/Substr/pos:0
1decode_image/cond_jpeg/cond_png/cond_gif/Substr:0
1decode_image/cond_jpeg/cond_png/cond_gif/Switch:0
1decode_image/cond_jpeg/cond_png/cond_gif/Switch:1
;decode_image/cond_jpeg/cond_png/cond_gif/check_channels/x:0
;decode_image/cond_jpeg/cond_png/cond_gif/check_channels/y:0
9decode_image/cond_jpeg/cond_png/cond_gif/check_channels:0
?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/x:0
?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/y:0
=decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels:0
Adecode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/x:0
Adecode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/y:0
?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1:0
3decode_image/cond_jpeg/cond_png/cond_gif/is_bmp/y:0
1decode_image/cond_jpeg/cond_png/cond_gif/is_bmp:0
2decode_image/cond_jpeg/cond_png/cond_gif/pred_id:0
3decode_image/cond_jpeg/cond_png/cond_gif/switch_f:0
3decode_image/cond_jpeg/cond_png/cond_gif/switch_t:0
/decode_image/cond_jpeg/cond_png/is_gif/Switch:0
1decode_image/cond_jpeg/cond_png/is_gif/Switch_1:0
*decode_image/cond_jpeg/cond_png/is_gif/y:0
(decode_image/cond_jpeg/cond_png/is_gif:0
)decode_image/cond_jpeg/cond_png/pred_id:0
*decode_image/cond_jpeg/cond_png/switch_f:0
*decode_image/cond_jpeg/cond_png/switch_t:0
'decode_image/cond_jpeg/is_png/Equal/y:0
%decode_image/cond_jpeg/is_png/Equal:0
-decode_image/cond_jpeg/is_png/Substr/Switch:0
*decode_image/cond_jpeg/is_png/Substr/len:0
*decode_image/cond_jpeg/is_png/Substr/pos:0
&decode_image/cond_jpeg/is_png/Substr:0
 decode_image/cond_jpeg/pred_id:0
!decode_image/cond_jpeg/switch_f:0
x:04
x:0-decode_image/cond_jpeg/is_png/Substr/Switch:0D
 decode_image/cond_jpeg/pred_id:0 decode_image/cond_jpeg/pred_id:0H
decode_image/Substr:0/decode_image/cond_jpeg/cond_png/is_gif/Switch:02�
�
)decode_image/cond_jpeg/cond_png/cond_text)decode_image/cond_jpeg/cond_png/pred_id:0*decode_image/cond_jpeg/cond_png/switch_t:0 *�
2decode_image/cond_jpeg/cond_png/DecodePng/Switch:1
+decode_image/cond_jpeg/cond_png/DecodePng:0
*decode_image/cond_jpeg/cond_png/Identity:0
)decode_image/cond_jpeg/cond_png/pred_id:0
*decode_image/cond_jpeg/cond_png/switch_t:0
-decode_image/cond_jpeg/is_png/Substr/Switch:0
x:09
x:02decode_image/cond_jpeg/cond_png/DecodePng/Switch:1^
-decode_image/cond_jpeg/is_png/Substr/Switch:0-decode_image/cond_jpeg/is_png/Substr/Switch:0V
)decode_image/cond_jpeg/cond_png/pred_id:0)decode_image/cond_jpeg/cond_png/pred_id:02�-
�-
+decode_image/cond_jpeg/cond_png/cond_text_1)decode_image/cond_jpeg/cond_png/pred_id:0*decode_image/cond_jpeg/cond_png/switch_f:0*�
decode_image/Substr:0
?decode_image/cond_jpeg/cond_png/cond_gif/Assert/Assert/data_0:0
7decode_image/cond_jpeg/cond_png/cond_gif/Assert/Const:0
Adecode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Assert/data_0:0
9decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Const:0
Adecode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Assert/data_0:0
9decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Const:0
4decode_image/cond_jpeg/cond_png/cond_gif/DecodeBmp:0
;decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch:0
=decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch_1:1
4decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif:0
3decode_image/cond_jpeg/cond_png/cond_gif/Identity:0
5decode_image/cond_jpeg/cond_png/cond_gif/Identity_1:0
5decode_image/cond_jpeg/cond_png/cond_gif/LogicalAnd:0
0decode_image/cond_jpeg/cond_png/cond_gif/Merge:0
0decode_image/cond_jpeg/cond_png/cond_gif/Merge:1
8decode_image/cond_jpeg/cond_png/cond_gif/Substr/Switch:0
5decode_image/cond_jpeg/cond_png/cond_gif/Substr/len:0
5decode_image/cond_jpeg/cond_png/cond_gif/Substr/pos:0
1decode_image/cond_jpeg/cond_png/cond_gif/Substr:0
1decode_image/cond_jpeg/cond_png/cond_gif/Switch:0
1decode_image/cond_jpeg/cond_png/cond_gif/Switch:1
;decode_image/cond_jpeg/cond_png/cond_gif/check_channels/x:0
;decode_image/cond_jpeg/cond_png/cond_gif/check_channels/y:0
9decode_image/cond_jpeg/cond_png/cond_gif/check_channels:0
?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/x:0
?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/y:0
=decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels:0
Adecode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/x:0
Adecode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/y:0
?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1:0
3decode_image/cond_jpeg/cond_png/cond_gif/is_bmp/y:0
1decode_image/cond_jpeg/cond_png/cond_gif/is_bmp:0
2decode_image/cond_jpeg/cond_png/cond_gif/pred_id:0
3decode_image/cond_jpeg/cond_png/cond_gif/switch_f:0
3decode_image/cond_jpeg/cond_png/cond_gif/switch_t:0
/decode_image/cond_jpeg/cond_png/is_gif/Switch:0
1decode_image/cond_jpeg/cond_png/is_gif/Switch_1:0
*decode_image/cond_jpeg/cond_png/is_gif/y:0
(decode_image/cond_jpeg/cond_png/is_gif:0
)decode_image/cond_jpeg/cond_png/pred_id:0
*decode_image/cond_jpeg/cond_png/switch_f:0
-decode_image/cond_jpeg/is_png/Substr/Switch:0
x:0B
x:0;decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch:0b
/decode_image/cond_jpeg/cond_png/is_gif/Switch:0/decode_image/cond_jpeg/cond_png/is_gif/Switch:0^
-decode_image/cond_jpeg/is_png/Substr/Switch:0-decode_image/cond_jpeg/is_png/Substr/Switch:0J
decode_image/Substr:01decode_image/cond_jpeg/cond_png/is_gif/Switch_1:0V
)decode_image/cond_jpeg/cond_png/pred_id:0)decode_image/cond_jpeg/cond_png/pred_id:02�

�

2decode_image/cond_jpeg/cond_png/cond_gif/cond_text2decode_image/cond_jpeg/cond_png/cond_gif/pred_id:03decode_image/cond_jpeg/cond_png/cond_gif/switch_t:0 *�	
?decode_image/cond_jpeg/cond_png/cond_gif/Assert/Assert/data_0:0
7decode_image/cond_jpeg/cond_png/cond_gif/Assert/Const:0
;decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch:0
=decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch_1:1
4decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif:0
3decode_image/cond_jpeg/cond_png/cond_gif/Identity:0
5decode_image/cond_jpeg/cond_png/cond_gif/LogicalAnd:0
?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/x:0
?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels/y:0
=decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels:0
Adecode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/x:0
Adecode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1/y:0
?decode_image/cond_jpeg/cond_png/cond_gif/check_gif_channels_1:0
2decode_image/cond_jpeg/cond_png/cond_gif/pred_id:0
3decode_image/cond_jpeg/cond_png/cond_gif/switch_t:0
x:0h
2decode_image/cond_jpeg/cond_png/cond_gif/pred_id:02decode_image/cond_jpeg/cond_png/cond_gif/pred_id:0D
x:0=decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch_1:1z
;decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch:0;decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch:02�
�
4decode_image/cond_jpeg/cond_png/cond_gif/cond_text_12decode_image/cond_jpeg/cond_png/cond_gif/pred_id:03decode_image/cond_jpeg/cond_png/cond_gif/switch_f:0*�

Adecode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Assert/data_0:0
9decode_image/cond_jpeg/cond_png/cond_gif/Assert_1/Const:0
Adecode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Assert/data_0:0
9decode_image/cond_jpeg/cond_png/cond_gif/Assert_2/Const:0
4decode_image/cond_jpeg/cond_png/cond_gif/DecodeBmp:0
;decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch:0
5decode_image/cond_jpeg/cond_png/cond_gif/Identity_1:0
8decode_image/cond_jpeg/cond_png/cond_gif/Substr/Switch:0
5decode_image/cond_jpeg/cond_png/cond_gif/Substr/len:0
5decode_image/cond_jpeg/cond_png/cond_gif/Substr/pos:0
1decode_image/cond_jpeg/cond_png/cond_gif/Substr:0
;decode_image/cond_jpeg/cond_png/cond_gif/check_channels/x:0
;decode_image/cond_jpeg/cond_png/cond_gif/check_channels/y:0
9decode_image/cond_jpeg/cond_png/cond_gif/check_channels:0
3decode_image/cond_jpeg/cond_png/cond_gif/is_bmp/y:0
1decode_image/cond_jpeg/cond_png/cond_gif/is_bmp:0
2decode_image/cond_jpeg/cond_png/cond_gif/pred_id:0
3decode_image/cond_jpeg/cond_png/cond_gif/switch_f:0
x:0h
2decode_image/cond_jpeg/cond_png/cond_gif/pred_id:02decode_image/cond_jpeg/cond_png/cond_gif/pred_id:0?
x:08decode_image/cond_jpeg/cond_png/cond_gif/Substr/Switch:0z
;decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch:0;decode_image/cond_jpeg/cond_png/cond_gif/DecodeGif/Switch:0"�6
while_context�6�6
�6
lstm/rnn/while/while_context *lstm/rnn/while/LoopCond:02lstm/rnn/while/Merge:0:lstm/rnn/while/Identity:0Blstm/rnn/while/Exit:0Blstm/rnn/while/Exit_1:0Blstm/rnn/while/Exit_2:0Blstm/rnn/while/Exit_3:0Blstm/rnn/while/Exit_4:0Blstm/rnn/while/Exit_5:0Blstm/rnn/while/Exit_6:0J�2
lstm/rnn/CheckSeqLen:0
lstm/rnn/Minimum:0
lstm/rnn/TensorArray:0
Elstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
lstm/rnn/TensorArray_1:0
4lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0
6lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0
4lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0
6lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0
lstm/rnn/strided_slice:0
lstm/rnn/while/Enter:0
lstm/rnn/while/Enter_1:0
lstm/rnn/while/Enter_2:0
lstm/rnn/while/Enter_3:0
lstm/rnn/while/Enter_4:0
lstm/rnn/while/Enter_5:0
lstm/rnn/while/Enter_6:0
lstm/rnn/while/Exit:0
lstm/rnn/while/Exit_1:0
lstm/rnn/while/Exit_2:0
lstm/rnn/while/Exit_3:0
lstm/rnn/while/Exit_4:0
lstm/rnn/while/Exit_5:0
lstm/rnn/while/Exit_6:0
#lstm/rnn/while/GreaterEqual/Enter:0
lstm/rnn/while/GreaterEqual:0
lstm/rnn/while/Identity:0
lstm/rnn/while/Identity_1:0
lstm/rnn/while/Identity_2:0
lstm/rnn/while/Identity_3:0
lstm/rnn/while/Identity_4:0
lstm/rnn/while/Identity_5:0
lstm/rnn/while/Identity_6:0
lstm/rnn/while/Less/Enter:0
lstm/rnn/while/Less:0
lstm/rnn/while/Less_1/Enter:0
lstm/rnn/while/Less_1:0
lstm/rnn/while/LogicalAnd:0
lstm/rnn/while/LoopCond:0
lstm/rnn/while/Merge:0
lstm/rnn/while/Merge:1
lstm/rnn/while/Merge_1:0
lstm/rnn/while/Merge_1:1
lstm/rnn/while/Merge_2:0
lstm/rnn/while/Merge_2:1
lstm/rnn/while/Merge_3:0
lstm/rnn/while/Merge_3:1
lstm/rnn/while/Merge_4:0
lstm/rnn/while/Merge_4:1
lstm/rnn/while/Merge_5:0
lstm/rnn/while/Merge_5:1
lstm/rnn/while/Merge_6:0
lstm/rnn/while/Merge_6:1
lstm/rnn/while/NextIteration:0
 lstm/rnn/while/NextIteration_1:0
 lstm/rnn/while/NextIteration_2:0
 lstm/rnn/while/NextIteration_3:0
 lstm/rnn/while/NextIteration_4:0
 lstm/rnn/while/NextIteration_5:0
 lstm/rnn/while/NextIteration_6:0
lstm/rnn/while/Select/Enter:0
lstm/rnn/while/Select:0
lstm/rnn/while/Select_1:0
lstm/rnn/while/Select_2:0
lstm/rnn/while/Select_3:0
lstm/rnn/while/Select_4:0
lstm/rnn/while/Switch:0
lstm/rnn/while/Switch:1
lstm/rnn/while/Switch_1:0
lstm/rnn/while/Switch_1:1
lstm/rnn/while/Switch_2:0
lstm/rnn/while/Switch_2:1
lstm/rnn/while/Switch_3:0
lstm/rnn/while/Switch_3:1
lstm/rnn/while/Switch_4:0
lstm/rnn/while/Switch_4:1
lstm/rnn/while/Switch_5:0
lstm/rnn/while/Switch_5:1
lstm/rnn/while/Switch_6:0
lstm/rnn/while/Switch_6:1
(lstm/rnn/while/TensorArrayReadV3/Enter:0
*lstm/rnn/while/TensorArrayReadV3/Enter_1:0
"lstm/rnn/while/TensorArrayReadV3:0
:lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
4lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
lstm/rnn/while/add/y:0
lstm/rnn/while/add:0
lstm/rnn/while/add_1/y:0
lstm/rnn/while/add_1:0
Blstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter:0
<lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Const:0
Alstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter:0
;lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul:0
<lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid:0
>lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1:0
>lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2:0
9lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh:0
;lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/y:0
8lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1:0
@lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axis:0
;lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat:0
8lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2:0
Dlstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dim:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:1
:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:2
:lstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:3
Blstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter:0
<lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Const:0
Alstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter:0
;lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul:0
<lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid:0
>lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1:0
>lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2:0
9lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh:0
;lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add/y:0
8lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1:0
@lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat/axis:0
;lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat:0
8lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2:0
Dlstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split/split_dim:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:0
:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:1
:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:2
:lstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:3
lstm/rnn/zeros:0z
4lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0Blstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter:0T
lstm/rnn/TensorArray:0:lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:07
lstm/rnn/strided_slice:0lstm/rnn/while/Less/Enter:0s
Elstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0*lstm/rnn/while/TensorArrayReadV3/Enter_1:03
lstm/rnn/Minimum:0lstm/rnn/while/Less_1/Enter:0{
6lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0Alstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter:01
lstm/rnn/zeros:0lstm/rnn/while/Select/Enter:0{
6lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0Alstm/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter:0z
4lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0Blstm/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter:0=
lstm/rnn/CheckSeqLen:0#lstm/rnn/while/GreaterEqual/Enter:0D
lstm/rnn/TensorArray_1:0(lstm/rnn/while/TensorArrayReadV3/Enter:0Rlstm/rnn/while/Enter:0Rlstm/rnn/while/Enter_1:0Rlstm/rnn/while/Enter_2:0Rlstm/rnn/while/Enter_3:0Rlstm/rnn/while/Enter_4:0Rlstm/rnn/while/Enter_5:0Rlstm/rnn/while/Enter_6:0Zlstm/rnn/strided_slice:0"�7
	variables�7�7
�
cnn/layer-conv1/weight:0cnn/layer-conv1/weight/Assigncnn/layer-conv1/weight/read:025cnn/layer-conv1/weight/Initializer/truncated_normal:08
~
cnn/layer-conv1/bias:0cnn/layer-conv1/bias/Assigncnn/layer-conv1/bias/read:02(cnn/layer-conv1/bias/Initializer/Const:08
�
*cnn/layer-conv1/batch-normalization/beta:0/cnn/layer-conv1/batch-normalization/beta/Assign/cnn/layer-conv1/batch-normalization/beta/read:02<cnn/layer-conv1/batch-normalization/beta/Initializer/zeros:08
�
+cnn/layer-conv1/batch-normalization/gamma:00cnn/layer-conv1/batch-normalization/gamma/Assign0cnn/layer-conv1/batch-normalization/gamma/read:02<cnn/layer-conv1/batch-normalization/gamma/Initializer/ones:08
�
1cnn/layer-conv1/batch-normalization/moving_mean:06cnn/layer-conv1/batch-normalization/moving_mean/Assign6cnn/layer-conv1/batch-normalization/moving_mean/read:02Ccnn/layer-conv1/batch-normalization/moving_mean/Initializer/zeros:0
�
5cnn/layer-conv1/batch-normalization/moving_variance:0:cnn/layer-conv1/batch-normalization/moving_variance/Assign:cnn/layer-conv1/batch-normalization/moving_variance/read:02Fcnn/layer-conv1/batch-normalization/moving_variance/Initializer/ones:0
�
cnn/layer-conv1-1/weight:0cnn/layer-conv1-1/weight/Assigncnn/layer-conv1-1/weight/read:027cnn/layer-conv1-1/weight/Initializer/truncated_normal:08
�
cnn/layer-conv1-1/bias:0cnn/layer-conv1-1/bias/Assigncnn/layer-conv1-1/bias/read:02*cnn/layer-conv1-1/bias/Initializer/Const:08
�
,cnn/layer-conv1-1/batch-normalization/beta:01cnn/layer-conv1-1/batch-normalization/beta/Assign1cnn/layer-conv1-1/batch-normalization/beta/read:02>cnn/layer-conv1-1/batch-normalization/beta/Initializer/zeros:08
�
-cnn/layer-conv1-1/batch-normalization/gamma:02cnn/layer-conv1-1/batch-normalization/gamma/Assign2cnn/layer-conv1-1/batch-normalization/gamma/read:02>cnn/layer-conv1-1/batch-normalization/gamma/Initializer/ones:08
�
3cnn/layer-conv1-1/batch-normalization/moving_mean:08cnn/layer-conv1-1/batch-normalization/moving_mean/Assign8cnn/layer-conv1-1/batch-normalization/moving_mean/read:02Ecnn/layer-conv1-1/batch-normalization/moving_mean/Initializer/zeros:0
�
7cnn/layer-conv1-1/batch-normalization/moving_variance:0<cnn/layer-conv1-1/batch-normalization/moving_variance/Assign<cnn/layer-conv1-1/batch-normalization/moving_variance/read:02Hcnn/layer-conv1-1/batch-normalization/moving_variance/Initializer/ones:0
�
cnn/layer-conv2/weight:0cnn/layer-conv2/weight/Assigncnn/layer-conv2/weight/read:025cnn/layer-conv2/weight/Initializer/truncated_normal:08
~
cnn/layer-conv2/bias:0cnn/layer-conv2/bias/Assigncnn/layer-conv2/bias/read:02(cnn/layer-conv2/bias/Initializer/Const:08
�
*cnn/layer-conv2/batch-normalization/beta:0/cnn/layer-conv2/batch-normalization/beta/Assign/cnn/layer-conv2/batch-normalization/beta/read:02<cnn/layer-conv2/batch-normalization/beta/Initializer/zeros:08
�
+cnn/layer-conv2/batch-normalization/gamma:00cnn/layer-conv2/batch-normalization/gamma/Assign0cnn/layer-conv2/batch-normalization/gamma/read:02<cnn/layer-conv2/batch-normalization/gamma/Initializer/ones:08
�
1cnn/layer-conv2/batch-normalization/moving_mean:06cnn/layer-conv2/batch-normalization/moving_mean/Assign6cnn/layer-conv2/batch-normalization/moving_mean/read:02Ccnn/layer-conv2/batch-normalization/moving_mean/Initializer/zeros:0
�
5cnn/layer-conv2/batch-normalization/moving_variance:0:cnn/layer-conv2/batch-normalization/moving_variance/Assign:cnn/layer-conv2/batch-normalization/moving_variance/read:02Fcnn/layer-conv2/batch-normalization/moving_variance/Initializer/ones:0
�
cnn/layer-conv3/weight:0cnn/layer-conv3/weight/Assigncnn/layer-conv3/weight/read:025cnn/layer-conv3/weight/Initializer/truncated_normal:08
~
cnn/layer-conv3/bias:0cnn/layer-conv3/bias/Assigncnn/layer-conv3/bias/read:02(cnn/layer-conv3/bias/Initializer/Const:08
�
*cnn/layer-conv3/batch-normalization/beta:0/cnn/layer-conv3/batch-normalization/beta/Assign/cnn/layer-conv3/batch-normalization/beta/read:02<cnn/layer-conv3/batch-normalization/beta/Initializer/zeros:08
�
+cnn/layer-conv3/batch-normalization/gamma:00cnn/layer-conv3/batch-normalization/gamma/Assign0cnn/layer-conv3/batch-normalization/gamma/read:02<cnn/layer-conv3/batch-normalization/gamma/Initializer/ones:08
�
1cnn/layer-conv3/batch-normalization/moving_mean:06cnn/layer-conv3/batch-normalization/moving_mean/Assign6cnn/layer-conv3/batch-normalization/moving_mean/read:02Ccnn/layer-conv3/batch-normalization/moving_mean/Initializer/zeros:0
�
5cnn/layer-conv3/batch-normalization/moving_variance:0:cnn/layer-conv3/batch-normalization/moving_variance/Assign:cnn/layer-conv3/batch-normalization/moving_variance/read:02Fcnn/layer-conv3/batch-normalization/moving_variance/Initializer/ones:0
�
cnn/layer-conv3-1/weight:0cnn/layer-conv3-1/weight/Assigncnn/layer-conv3-1/weight/read:027cnn/layer-conv3-1/weight/Initializer/truncated_normal:08
�
cnn/layer-conv3-1/bias:0cnn/layer-conv3-1/bias/Assigncnn/layer-conv3-1/bias/read:02*cnn/layer-conv3-1/bias/Initializer/Const:08
�
,cnn/layer-conv3-1/batch-normalization/beta:01cnn/layer-conv3-1/batch-normalization/beta/Assign1cnn/layer-conv3-1/batch-normalization/beta/read:02>cnn/layer-conv3-1/batch-normalization/beta/Initializer/zeros:08
�
-cnn/layer-conv3-1/batch-normalization/gamma:02cnn/layer-conv3-1/batch-normalization/gamma/Assign2cnn/layer-conv3-1/batch-normalization/gamma/read:02>cnn/layer-conv3-1/batch-normalization/gamma/Initializer/ones:08
�
3cnn/layer-conv3-1/batch-normalization/moving_mean:08cnn/layer-conv3-1/batch-normalization/moving_mean/Assign8cnn/layer-conv3-1/batch-normalization/moving_mean/read:02Ecnn/layer-conv3-1/batch-normalization/moving_mean/Initializer/zeros:0
�
7cnn/layer-conv3-1/batch-normalization/moving_variance:0<cnn/layer-conv3-1/batch-normalization/moving_variance/Assign<cnn/layer-conv3-1/batch-normalization/moving_variance/read:02Hcnn/layer-conv3-1/batch-normalization/moving_variance/Initializer/ones:0
�
1lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:06lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign6lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:02Llstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform:08
�
/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:04lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign4lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:02Alstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros:08
�
1lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:06lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign6lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:02Llstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform:08
�
/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:04lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign4lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:02Alstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zeros:08
e
lstm/weight:0lstm/weight/Assignlstm/weight/read:02*lstm/weight/Initializer/truncated_normal:08
R
lstm/bias:0lstm/bias/Assignlstm/bias/read:02lstm/bias/Initializer/Const:08"�$
model_variables�#�#
�
*cnn/layer-conv1/batch-normalization/beta:0/cnn/layer-conv1/batch-normalization/beta/Assign/cnn/layer-conv1/batch-normalization/beta/read:02<cnn/layer-conv1/batch-normalization/beta/Initializer/zeros:08
�
+cnn/layer-conv1/batch-normalization/gamma:00cnn/layer-conv1/batch-normalization/gamma/Assign0cnn/layer-conv1/batch-normalization/gamma/read:02<cnn/layer-conv1/batch-normalization/gamma/Initializer/ones:08
�
1cnn/layer-conv1/batch-normalization/moving_mean:06cnn/layer-conv1/batch-normalization/moving_mean/Assign6cnn/layer-conv1/batch-normalization/moving_mean/read:02Ccnn/layer-conv1/batch-normalization/moving_mean/Initializer/zeros:0
�
5cnn/layer-conv1/batch-normalization/moving_variance:0:cnn/layer-conv1/batch-normalization/moving_variance/Assign:cnn/layer-conv1/batch-normalization/moving_variance/read:02Fcnn/layer-conv1/batch-normalization/moving_variance/Initializer/ones:0
�
,cnn/layer-conv1-1/batch-normalization/beta:01cnn/layer-conv1-1/batch-normalization/beta/Assign1cnn/layer-conv1-1/batch-normalization/beta/read:02>cnn/layer-conv1-1/batch-normalization/beta/Initializer/zeros:08
�
-cnn/layer-conv1-1/batch-normalization/gamma:02cnn/layer-conv1-1/batch-normalization/gamma/Assign2cnn/layer-conv1-1/batch-normalization/gamma/read:02>cnn/layer-conv1-1/batch-normalization/gamma/Initializer/ones:08
�
3cnn/layer-conv1-1/batch-normalization/moving_mean:08cnn/layer-conv1-1/batch-normalization/moving_mean/Assign8cnn/layer-conv1-1/batch-normalization/moving_mean/read:02Ecnn/layer-conv1-1/batch-normalization/moving_mean/Initializer/zeros:0
�
7cnn/layer-conv1-1/batch-normalization/moving_variance:0<cnn/layer-conv1-1/batch-normalization/moving_variance/Assign<cnn/layer-conv1-1/batch-normalization/moving_variance/read:02Hcnn/layer-conv1-1/batch-normalization/moving_variance/Initializer/ones:0
�
*cnn/layer-conv2/batch-normalization/beta:0/cnn/layer-conv2/batch-normalization/beta/Assign/cnn/layer-conv2/batch-normalization/beta/read:02<cnn/layer-conv2/batch-normalization/beta/Initializer/zeros:08
�
+cnn/layer-conv2/batch-normalization/gamma:00cnn/layer-conv2/batch-normalization/gamma/Assign0cnn/layer-conv2/batch-normalization/gamma/read:02<cnn/layer-conv2/batch-normalization/gamma/Initializer/ones:08
�
1cnn/layer-conv2/batch-normalization/moving_mean:06cnn/layer-conv2/batch-normalization/moving_mean/Assign6cnn/layer-conv2/batch-normalization/moving_mean/read:02Ccnn/layer-conv2/batch-normalization/moving_mean/Initializer/zeros:0
�
5cnn/layer-conv2/batch-normalization/moving_variance:0:cnn/layer-conv2/batch-normalization/moving_variance/Assign:cnn/layer-conv2/batch-normalization/moving_variance/read:02Fcnn/layer-conv2/batch-normalization/moving_variance/Initializer/ones:0
�
*cnn/layer-conv3/batch-normalization/beta:0/cnn/layer-conv3/batch-normalization/beta/Assign/cnn/layer-conv3/batch-normalization/beta/read:02<cnn/layer-conv3/batch-normalization/beta/Initializer/zeros:08
�
+cnn/layer-conv3/batch-normalization/gamma:00cnn/layer-conv3/batch-normalization/gamma/Assign0cnn/layer-conv3/batch-normalization/gamma/read:02<cnn/layer-conv3/batch-normalization/gamma/Initializer/ones:08
�
1cnn/layer-conv3/batch-normalization/moving_mean:06cnn/layer-conv3/batch-normalization/moving_mean/Assign6cnn/layer-conv3/batch-normalization/moving_mean/read:02Ccnn/layer-conv3/batch-normalization/moving_mean/Initializer/zeros:0
�
5cnn/layer-conv3/batch-normalization/moving_variance:0:cnn/layer-conv3/batch-normalization/moving_variance/Assign:cnn/layer-conv3/batch-normalization/moving_variance/read:02Fcnn/layer-conv3/batch-normalization/moving_variance/Initializer/ones:0
�
,cnn/layer-conv3-1/batch-normalization/beta:01cnn/layer-conv3-1/batch-normalization/beta/Assign1cnn/layer-conv3-1/batch-normalization/beta/read:02>cnn/layer-conv3-1/batch-normalization/beta/Initializer/zeros:08
�
-cnn/layer-conv3-1/batch-normalization/gamma:02cnn/layer-conv3-1/batch-normalization/gamma/Assign2cnn/layer-conv3-1/batch-normalization/gamma/read:02>cnn/layer-conv3-1/batch-normalization/gamma/Initializer/ones:08
�
3cnn/layer-conv3-1/batch-normalization/moving_mean:08cnn/layer-conv3-1/batch-normalization/moving_mean/Assign8cnn/layer-conv3-1/batch-normalization/moving_mean/read:02Ecnn/layer-conv3-1/batch-normalization/moving_mean/Initializer/zeros:0
�
7cnn/layer-conv3-1/batch-normalization/moving_variance:0<cnn/layer-conv3-1/batch-normalization/moving_variance/Assign<cnn/layer-conv3-1/batch-normalization/moving_variance/read:02Hcnn/layer-conv3-1/batch-normalization/moving_variance/Initializer/ones:0"�$
trainable_variables�$�$
�
cnn/layer-conv1/weight:0cnn/layer-conv1/weight/Assigncnn/layer-conv1/weight/read:025cnn/layer-conv1/weight/Initializer/truncated_normal:08
~
cnn/layer-conv1/bias:0cnn/layer-conv1/bias/Assigncnn/layer-conv1/bias/read:02(cnn/layer-conv1/bias/Initializer/Const:08
�
*cnn/layer-conv1/batch-normalization/beta:0/cnn/layer-conv1/batch-normalization/beta/Assign/cnn/layer-conv1/batch-normalization/beta/read:02<cnn/layer-conv1/batch-normalization/beta/Initializer/zeros:08
�
+cnn/layer-conv1/batch-normalization/gamma:00cnn/layer-conv1/batch-normalization/gamma/Assign0cnn/layer-conv1/batch-normalization/gamma/read:02<cnn/layer-conv1/batch-normalization/gamma/Initializer/ones:08
�
cnn/layer-conv1-1/weight:0cnn/layer-conv1-1/weight/Assigncnn/layer-conv1-1/weight/read:027cnn/layer-conv1-1/weight/Initializer/truncated_normal:08
�
cnn/layer-conv1-1/bias:0cnn/layer-conv1-1/bias/Assigncnn/layer-conv1-1/bias/read:02*cnn/layer-conv1-1/bias/Initializer/Const:08
�
,cnn/layer-conv1-1/batch-normalization/beta:01cnn/layer-conv1-1/batch-normalization/beta/Assign1cnn/layer-conv1-1/batch-normalization/beta/read:02>cnn/layer-conv1-1/batch-normalization/beta/Initializer/zeros:08
�
-cnn/layer-conv1-1/batch-normalization/gamma:02cnn/layer-conv1-1/batch-normalization/gamma/Assign2cnn/layer-conv1-1/batch-normalization/gamma/read:02>cnn/layer-conv1-1/batch-normalization/gamma/Initializer/ones:08
�
cnn/layer-conv2/weight:0cnn/layer-conv2/weight/Assigncnn/layer-conv2/weight/read:025cnn/layer-conv2/weight/Initializer/truncated_normal:08
~
cnn/layer-conv2/bias:0cnn/layer-conv2/bias/Assigncnn/layer-conv2/bias/read:02(cnn/layer-conv2/bias/Initializer/Const:08
�
*cnn/layer-conv2/batch-normalization/beta:0/cnn/layer-conv2/batch-normalization/beta/Assign/cnn/layer-conv2/batch-normalization/beta/read:02<cnn/layer-conv2/batch-normalization/beta/Initializer/zeros:08
�
+cnn/layer-conv2/batch-normalization/gamma:00cnn/layer-conv2/batch-normalization/gamma/Assign0cnn/layer-conv2/batch-normalization/gamma/read:02<cnn/layer-conv2/batch-normalization/gamma/Initializer/ones:08
�
cnn/layer-conv3/weight:0cnn/layer-conv3/weight/Assigncnn/layer-conv3/weight/read:025cnn/layer-conv3/weight/Initializer/truncated_normal:08
~
cnn/layer-conv3/bias:0cnn/layer-conv3/bias/Assigncnn/layer-conv3/bias/read:02(cnn/layer-conv3/bias/Initializer/Const:08
�
*cnn/layer-conv3/batch-normalization/beta:0/cnn/layer-conv3/batch-normalization/beta/Assign/cnn/layer-conv3/batch-normalization/beta/read:02<cnn/layer-conv3/batch-normalization/beta/Initializer/zeros:08
�
+cnn/layer-conv3/batch-normalization/gamma:00cnn/layer-conv3/batch-normalization/gamma/Assign0cnn/layer-conv3/batch-normalization/gamma/read:02<cnn/layer-conv3/batch-normalization/gamma/Initializer/ones:08
�
cnn/layer-conv3-1/weight:0cnn/layer-conv3-1/weight/Assigncnn/layer-conv3-1/weight/read:027cnn/layer-conv3-1/weight/Initializer/truncated_normal:08
�
cnn/layer-conv3-1/bias:0cnn/layer-conv3-1/bias/Assigncnn/layer-conv3-1/bias/read:02*cnn/layer-conv3-1/bias/Initializer/Const:08
�
,cnn/layer-conv3-1/batch-normalization/beta:01cnn/layer-conv3-1/batch-normalization/beta/Assign1cnn/layer-conv3-1/batch-normalization/beta/read:02>cnn/layer-conv3-1/batch-normalization/beta/Initializer/zeros:08
�
-cnn/layer-conv3-1/batch-normalization/gamma:02cnn/layer-conv3-1/batch-normalization/gamma/Assign2cnn/layer-conv3-1/batch-normalization/gamma/read:02>cnn/layer-conv3-1/batch-normalization/gamma/Initializer/ones:08
�
1lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:06lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign6lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:02Llstm/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform:08
�
/lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:04lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign4lstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:02Alstm/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros:08
�
1lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:06lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign6lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:02Llstm/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform:08
�
/lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:04lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign4lstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:02Alstm/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zeros:08
e
lstm/weight:0lstm/weight/Assignlstm/weight/read:02*lstm/weight/Initializer/truncated_normal:08
R
lstm/bias:0lstm/bias/Assignlstm/bias/read:02lstm/bias/Initializer/Const:08*x
serving_defaulte

image
x:01

prediction#
y:0	������������������tensorflow/serving/predict