       �K"	  @I���Abrain.Event:2�Vh��     diY	�uHI���A"��
�
input/imagesPlaceholder"/device:GPU:0*/
_output_shapes
:���������@@*$
shape:���������@@*
dtype0
�
input/correct_labelsPlaceholder"/device:GPU:0*
shape:���������*
dtype0*'
_output_shapes
:���������
e
input/PlaceholderPlaceholder"/device:GPU:0*
shape:*
dtype0
*
_output_shapes
:
�
#CNN1/weights/truncated_normal/shapeConst"/device:GPU:0*
_output_shapes
:*%
valueB"            *
dtype0
v
"CNN1/weights/truncated_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
x
$CNN1/weights/truncated_normal/stddevConst"/device:GPU:0*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
-CNN1/weights/truncated_normal/TruncatedNormalTruncatedNormal#CNN1/weights/truncated_normal/shape"/device:GPU:0*
T0*
dtype0*&
_output_shapes
:*
seed2 *

seed 
�
!CNN1/weights/truncated_normal/mulMul-CNN1/weights/truncated_normal/TruncatedNormal$CNN1/weights/truncated_normal/stddev"/device:GPU:0*&
_output_shapes
:*
T0
�
CNN1/weights/truncated_normalAdd!CNN1/weights/truncated_normal/mul"CNN1/weights/truncated_normal/mean"/device:GPU:0*
T0*&
_output_shapes
:
�
CNN1/weights/Variable
VariableV2"/device:GPU:0*&
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
CNN1/weights/Variable/AssignAssignCNN1/weights/VariableCNN1/weights/truncated_normal"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN1/weights/Variable*
validate_shape(*&
_output_shapes
:
�
CNN1/weights/Variable/readIdentityCNN1/weights/Variable"/device:GPU:0*&
_output_shapes
:*
T0*(
_class
loc:@CNN1/weights/Variable
l
CNN1/weights/summaries/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
s
"CNN1/weights/summaries/range/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
s
"CNN1/weights/summaries/range/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN1/weights/summaries/rangeRange"CNN1/weights/summaries/range/startCNN1/weights/summaries/Rank"CNN1/weights/summaries/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/weights/summaries/MeanMeanCNN1/weights/Variable/readCNN1/weights/summaries/range"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
 CNN1/weights/summaries/mean/tagsConst"/device:GPU:0*,
value#B! BCNN1/weights/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN1/weights/summaries/meanScalarSummary CNN1/weights/summaries/mean/tagsCNN1/weights/summaries/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
!CNN1/weights/summaries/stddev/subSubCNN1/weights/Variable/readCNN1/weights/summaries/Mean"/device:GPU:0*&
_output_shapes
:*
T0
�
$CNN1/weights/summaries/stddev/SquareSquare!CNN1/weights/summaries/stddev/sub"/device:GPU:0*&
_output_shapes
:*
T0
�
#CNN1/weights/summaries/stddev/ConstConst"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
"CNN1/weights/summaries/stddev/MeanMean$CNN1/weights/summaries/stddev/Square#CNN1/weights/summaries/stddev/Const"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
~
"CNN1/weights/summaries/stddev/SqrtSqrt"CNN1/weights/summaries/stddev/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
$CNN1/weights/summaries/stddev_1/tagsConst"/device:GPU:0*0
value'B% BCNN1/weights/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
CNN1/weights/summaries/stddev_1ScalarSummary$CNN1/weights/summaries/stddev_1/tags"CNN1/weights/summaries/stddev/Sqrt"/device:GPU:0*
_output_shapes
: *
T0
n
CNN1/weights/summaries/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
u
$CNN1/weights/summaries/range_1/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
u
$CNN1/weights/summaries/range_1/deltaConst"/device:GPU:0*
dtype0*
_output_shapes
: *
value	B :
�
CNN1/weights/summaries/range_1Range$CNN1/weights/summaries/range_1/startCNN1/weights/summaries/Rank_1$CNN1/weights/summaries/range_1/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
�
CNN1/weights/summaries/MaxMaxCNN1/weights/Variable/readCNN1/weights/summaries/range_1"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
CNN1/weights/summaries/max/tagsConst"/device:GPU:0*+
value"B  BCNN1/weights/summaries/max*
dtype0*
_output_shapes
: 
�
CNN1/weights/summaries/maxScalarSummaryCNN1/weights/summaries/max/tagsCNN1/weights/summaries/Max"/device:GPU:0*
_output_shapes
: *
T0
n
CNN1/weights/summaries/Rank_2Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
u
$CNN1/weights/summaries/range_2/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
u
$CNN1/weights/summaries/range_2/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN1/weights/summaries/range_2Range$CNN1/weights/summaries/range_2/startCNN1/weights/summaries/Rank_2$CNN1/weights/summaries/range_2/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/weights/summaries/MinMinCNN1/weights/Variable/readCNN1/weights/summaries/range_2"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN1/weights/summaries/min/tagsConst"/device:GPU:0*
_output_shapes
: *+
value"B  BCNN1/weights/summaries/min*
dtype0
�
CNN1/weights/summaries/minScalarSummaryCNN1/weights/summaries/min/tagsCNN1/weights/summaries/Min"/device:GPU:0*
_output_shapes
: *
T0
�
$CNN1/weights/summaries/histogram/tagConst"/device:GPU:0*1
value(B& B CNN1/weights/summaries/histogram*
dtype0*
_output_shapes
: 
�
 CNN1/weights/summaries/histogramHistogramSummary$CNN1/weights/summaries/histogram/tagCNN1/weights/Variable/read"/device:GPU:0*
_output_shapes
: *
T0
m
CNN1/biases/ConstConst"/device:GPU:0*
valueB*���=*
dtype0*
_output_shapes
:
�
CNN1/biases/Variable
VariableV2"/device:GPU:0*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
CNN1/biases/Variable/AssignAssignCNN1/biases/VariableCNN1/biases/Const"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
:
�
CNN1/biases/Variable/readIdentityCNN1/biases/Variable"/device:GPU:0*
T0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
:
k
CNN1/biases/summaries/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
r
!CNN1/biases/summaries/range/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
r
!CNN1/biases/summaries/range/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/rangeRange!CNN1/biases/summaries/range/startCNN1/biases/summaries/Rank!CNN1/biases/summaries/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/biases/summaries/MeanMeanCNN1/biases/Variable/readCNN1/biases/summaries/range"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN1/biases/summaries/mean/tagsConst"/device:GPU:0*+
value"B  BCNN1/biases/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/meanScalarSummaryCNN1/biases/summaries/mean/tagsCNN1/biases/summaries/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
 CNN1/biases/summaries/stddev/subSubCNN1/biases/Variable/readCNN1/biases/summaries/Mean"/device:GPU:0*
_output_shapes
:*
T0
�
#CNN1/biases/summaries/stddev/SquareSquare CNN1/biases/summaries/stddev/sub"/device:GPU:0*
_output_shapes
:*
T0
{
"CNN1/biases/summaries/stddev/ConstConst"/device:GPU:0*
_output_shapes
:*
valueB: *
dtype0
�
!CNN1/biases/summaries/stddev/MeanMean#CNN1/biases/summaries/stddev/Square"CNN1/biases/summaries/stddev/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
|
!CNN1/biases/summaries/stddev/SqrtSqrt!CNN1/biases/summaries/stddev/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
#CNN1/biases/summaries/stddev_1/tagsConst"/device:GPU:0*/
value&B$ BCNN1/biases/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/stddev_1ScalarSummary#CNN1/biases/summaries/stddev_1/tags!CNN1/biases/summaries/stddev/Sqrt"/device:GPU:0*
T0*
_output_shapes
: 
m
CNN1/biases/summaries/Rank_1Const"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
t
#CNN1/biases/summaries/range_1/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
t
#CNN1/biases/summaries/range_1/deltaConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
CNN1/biases/summaries/range_1Range#CNN1/biases/summaries/range_1/startCNN1/biases/summaries/Rank_1#CNN1/biases/summaries/range_1/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/biases/summaries/MaxMaxCNN1/biases/Variable/readCNN1/biases/summaries/range_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN1/biases/summaries/max/tagsConst"/device:GPU:0**
value!B BCNN1/biases/summaries/max*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/maxScalarSummaryCNN1/biases/summaries/max/tagsCNN1/biases/summaries/Max"/device:GPU:0*
_output_shapes
: *
T0
m
CNN1/biases/summaries/Rank_2Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
t
#CNN1/biases/summaries/range_2/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
t
#CNN1/biases/summaries/range_2/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/range_2Range#CNN1/biases/summaries/range_2/startCNN1/biases/summaries/Rank_2#CNN1/biases/summaries/range_2/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/biases/summaries/MinMinCNN1/biases/Variable/readCNN1/biases/summaries/range_2"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN1/biases/summaries/min/tagsConst"/device:GPU:0*
_output_shapes
: **
value!B BCNN1/biases/summaries/min*
dtype0
�
CNN1/biases/summaries/minScalarSummaryCNN1/biases/summaries/min/tagsCNN1/biases/summaries/Min"/device:GPU:0*
_output_shapes
: *
T0
�
#CNN1/biases/summaries/histogram/tagConst"/device:GPU:0*0
value'B% BCNN1/biases/summaries/histogram*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/histogramHistogramSummary#CNN1/biases/summaries/histogram/tagCNN1/biases/Variable/read"/device:GPU:0*
T0*
_output_shapes
: 
�
CNN1/Conv2DConv2Dinput/imagesCNN1/weights/Variable/read"/device:GPU:0*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������@@
�
CNN1/addAddCNN1/Conv2DCNN1/biases/Variable/read"/device:GPU:0*/
_output_shapes
:���������@@*
T0
d
	CNN1/ReluReluCNN1/add"/device:GPU:0*/
_output_shapes
:���������@@*
T0
t
CNN1/activations/tagConst"/device:GPU:0*!
valueB BCNN1/activations*
dtype0*
_output_shapes
: 
u
CNN1/activationsHistogramSummaryCNN1/activations/tag	CNN1/Relu"/device:GPU:0*
T0*
_output_shapes
: 
�
*batch_normalization/gamma/Initializer/onesConst*,
_class"
 loc:@batch_normalization/gamma*
valueB@*  �?*
dtype0*
_output_shapes
:@
�
batch_normalization/gamma
VariableV2"/device:GPU:0*,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
 batch_normalization/gamma/AssignAssignbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones"/device:GPU:0*
use_locking(*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:@
�
batch_normalization/gamma/readIdentitybatch_normalization/gamma"/device:GPU:0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@*
T0
�
*batch_normalization/beta/Initializer/zerosConst*+
_class!
loc:@batch_normalization/beta*
valueB@*    *
dtype0*
_output_shapes
:@
�
batch_normalization/beta
VariableV2"/device:GPU:0*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta
�
batch_normalization/beta/AssignAssignbatch_normalization/beta*batch_normalization/beta/Initializer/zeros"/device:GPU:0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
batch_normalization/beta/readIdentitybatch_normalization/beta"/device:GPU:0*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
�
1batch_normalization/moving_mean/Initializer/zerosConst*2
_class(
&$loc:@batch_normalization/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
�
batch_normalization/moving_mean
VariableV2"/device:GPU:0*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *2
_class(
&$loc:@batch_normalization/moving_mean*
	container 
�
&batch_normalization/moving_mean/AssignAssignbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
validate_shape(*
_output_shapes
:@
�
$batch_normalization/moving_mean/readIdentitybatch_normalization/moving_mean"/device:GPU:0*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
�
4batch_normalization/moving_variance/Initializer/onesConst*
_output_shapes
:@*6
_class,
*(loc:@batch_normalization/moving_variance*
valueB@*  �?*
dtype0
�
#batch_normalization/moving_variance
VariableV2"/device:GPU:0*
_output_shapes
:@*
shared_name *6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape:@*
dtype0
�
*batch_normalization/moving_variance/AssignAssign#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones"/device:GPU:0*
_output_shapes
:@*
use_locking(*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
validate_shape(
�
(batch_normalization/moving_variance/readIdentity#batch_normalization/moving_variance"/device:GPU:0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@*
T0
�
$CNN1/batch_normalization/cond/SwitchSwitchinput/Placeholderinput/Placeholder"/device:GPU:0*
_output_shapes

::*
T0

�
&CNN1/batch_normalization/cond/switch_tIdentity&CNN1/batch_normalization/cond/Switch:1"/device:GPU:0*
T0
*
_output_shapes
:
�
&CNN1/batch_normalization/cond/switch_fIdentity$CNN1/batch_normalization/cond/Switch"/device:GPU:0*
T0
*
_output_shapes
:
v
%CNN1/batch_normalization/cond/pred_idIdentityinput/Placeholder"/device:GPU:0*
_output_shapes
:*
T0

�
#CNN1/batch_normalization/cond/ConstConst'^CNN1/batch_normalization/cond/switch_t"/device:GPU:0*
_output_shapes
: *
valueB *
dtype0
�
%CNN1/batch_normalization/cond/Const_1Const'^CNN1/batch_normalization/cond/switch_t"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
3CNN1/batch_normalization/cond/FusedBatchNorm/SwitchSwitch	CNN1/Relu%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*J
_output_shapes8
6:���������@@:���������@@*
T0*
_class
loc:@CNN1/Relu
�
5CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1Switchbatch_normalization/gamma/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*
T0*,
_class"
 loc:@batch_normalization/gamma* 
_output_shapes
:@:@
�
5CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization/beta/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*
T0*+
_class!
loc:@batch_normalization/beta* 
_output_shapes
:@:@
�
,CNN1/batch_normalization/cond/FusedBatchNormFusedBatchNorm5CNN1/batch_normalization/cond/FusedBatchNorm/Switch:17CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1:17CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2:1#CNN1/batch_normalization/cond/Const%CNN1/batch_normalization/cond/Const_1"/device:GPU:0*
T0*
data_formatNCHW*G
_output_shapes5
3:���������@@:@:@:@:@*
is_training(*
epsilon%o�:
�
5CNN1/batch_normalization/cond/FusedBatchNorm_1/SwitchSwitch	CNN1/Relu%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*
_class
loc:@CNN1/Relu*J
_output_shapes8
6:���������@@:���������@@*
T0
�
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1Switchbatch_normalization/gamma/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*,
_class"
 loc:@batch_normalization/gamma* 
_output_shapes
:@:@*
T0
�
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization/beta/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
:@:@*
T0*+
_class!
loc:@batch_normalization/beta
�
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch$batch_normalization/moving_mean/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
:@:@*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch(batch_normalization/moving_variance/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*
T0*6
_class,
*(loc:@batch_normalization/moving_variance* 
_output_shapes
:@:@
�
.CNN1/batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm5CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_27CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
data_formatNCHW*G
_output_shapes5
3:���������@@:@:@:@:@*
is_training( *
epsilon%o�:*
T0
�
#CNN1/batch_normalization/cond/MergeMerge.CNN1/batch_normalization/cond/FusedBatchNorm_1,CNN1/batch_normalization/cond/FusedBatchNorm"/device:GPU:0*
T0*
N*1
_output_shapes
:���������@@: 
�
%CNN1/batch_normalization/cond/Merge_1Merge0CNN1/batch_normalization/cond/FusedBatchNorm_1:1.CNN1/batch_normalization/cond/FusedBatchNorm:1"/device:GPU:0*
N*
_output_shapes

:@: *
T0
�
%CNN1/batch_normalization/cond/Merge_2Merge0CNN1/batch_normalization/cond/FusedBatchNorm_1:2.CNN1/batch_normalization/cond/FusedBatchNorm:2"/device:GPU:0*
_output_shapes

:@: *
T0*
N
}
)CNN1/batch_normalization/ExpandDims/inputConst"/device:GPU:0*
_output_shapes
: *
valueB
 *
�#<*
dtype0
x
'CNN1/batch_normalization/ExpandDims/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
#CNN1/batch_normalization/ExpandDims
ExpandDims)CNN1/batch_normalization/ExpandDims/input'CNN1/batch_normalization/ExpandDims/dim"/device:GPU:0*
T0*
_output_shapes
:*

Tdim0

+CNN1/batch_normalization/ExpandDims_1/inputConst"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
z
)CNN1/batch_normalization/ExpandDims_1/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
%CNN1/batch_normalization/ExpandDims_1
ExpandDims+CNN1/batch_normalization/ExpandDims_1/input)CNN1/batch_normalization/ExpandDims_1/dim"/device:GPU:0*
_output_shapes
:*

Tdim0*
T0

&CNN1/batch_normalization/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
 CNN1/batch_normalization/ReshapeReshapeinput/Placeholder&CNN1/batch_normalization/Reshape/shape"/device:GPU:0*
_output_shapes
:*
T0
*
Tshape0
�
CNN1/batch_normalization/SelectSelect CNN1/batch_normalization/Reshape#CNN1/batch_normalization/ExpandDims%CNN1/batch_normalization/ExpandDims_1"/device:GPU:0*
_output_shapes
:*
T0
�
 CNN1/batch_normalization/SqueezeSqueezeCNN1/batch_normalization/Select"/device:GPU:0*
T0*
_output_shapes
: *
squeeze_dims
 
�
-CNN1/batch_normalization/AssignMovingAvg/readIdentitybatch_normalization/moving_mean"/device:GPU:0*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
�
,CNN1/batch_normalization/AssignMovingAvg/SubSub-CNN1/batch_normalization/AssignMovingAvg/read%CNN1/batch_normalization/cond/Merge_1"/device:GPU:0*
_output_shapes
:@*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
,CNN1/batch_normalization/AssignMovingAvg/MulMul,CNN1/batch_normalization/AssignMovingAvg/Sub CNN1/batch_normalization/Squeeze"/device:GPU:0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@*
T0
�
(CNN1/batch_normalization/AssignMovingAvg	AssignSubbatch_normalization/moving_mean,CNN1/batch_normalization/AssignMovingAvg/Mul"/device:GPU:0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@*
use_locking( *
T0
�
/CNN1/batch_normalization/AssignMovingAvg_1/readIdentity#batch_normalization/moving_variance"/device:GPU:0*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
�
.CNN1/batch_normalization/AssignMovingAvg_1/SubSub/CNN1/batch_normalization/AssignMovingAvg_1/read%CNN1/batch_normalization/cond/Merge_2"/device:GPU:0*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
�
.CNN1/batch_normalization/AssignMovingAvg_1/MulMul.CNN1/batch_normalization/AssignMovingAvg_1/Sub CNN1/batch_normalization/Squeeze"/device:GPU:0*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
�
*CNN1/batch_normalization/AssignMovingAvg_1	AssignSub#batch_normalization/moving_variance.CNN1/batch_normalization/AssignMovingAvg_1/Mul"/device:GPU:0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@*
use_locking( *
T0
r
CNN1/batch_norm/tagConst"/device:GPU:0* 
valueB BCNN1/batch_norm*
dtype0*
_output_shapes
: 
�
CNN1/batch_normHistogramSummaryCNN1/batch_norm/tag#CNN1/batch_normalization/cond/Merge"/device:GPU:0*
T0*
_output_shapes
: 
�
#CNN2/weights/truncated_normal/shapeConst"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
v
"CNN2/weights/truncated_normal/meanConst"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
x
$CNN2/weights/truncated_normal/stddevConst"/device:GPU:0*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
-CNN2/weights/truncated_normal/TruncatedNormalTruncatedNormal#CNN2/weights/truncated_normal/shape"/device:GPU:0*
dtype0*&
_output_shapes
: *
seed2 *

seed *
T0
�
!CNN2/weights/truncated_normal/mulMul-CNN2/weights/truncated_normal/TruncatedNormal$CNN2/weights/truncated_normal/stddev"/device:GPU:0*&
_output_shapes
: *
T0
�
CNN2/weights/truncated_normalAdd!CNN2/weights/truncated_normal/mul"CNN2/weights/truncated_normal/mean"/device:GPU:0*&
_output_shapes
: *
T0
�
CNN2/weights/Variable
VariableV2"/device:GPU:0*
shared_name *
dtype0*&
_output_shapes
: *
	container *
shape: 
�
CNN2/weights/Variable/AssignAssignCNN2/weights/VariableCNN2/weights/truncated_normal"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN2/weights/Variable*
validate_shape(*&
_output_shapes
: 
�
CNN2/weights/Variable/readIdentityCNN2/weights/Variable"/device:GPU:0*
T0*(
_class
loc:@CNN2/weights/Variable*&
_output_shapes
: 
l
CNN2/weights/summaries/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
s
"CNN2/weights/summaries/range/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
s
"CNN2/weights/summaries/range/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/rangeRange"CNN2/weights/summaries/range/startCNN2/weights/summaries/Rank"CNN2/weights/summaries/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN2/weights/summaries/MeanMeanCNN2/weights/Variable/readCNN2/weights/summaries/range"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
 CNN2/weights/summaries/mean/tagsConst"/device:GPU:0*,
value#B! BCNN2/weights/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/meanScalarSummary CNN2/weights/summaries/mean/tagsCNN2/weights/summaries/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
!CNN2/weights/summaries/stddev/subSubCNN2/weights/Variable/readCNN2/weights/summaries/Mean"/device:GPU:0*&
_output_shapes
: *
T0
�
$CNN2/weights/summaries/stddev/SquareSquare!CNN2/weights/summaries/stddev/sub"/device:GPU:0*
T0*&
_output_shapes
: 
�
#CNN2/weights/summaries/stddev/ConstConst"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
"CNN2/weights/summaries/stddev/MeanMean$CNN2/weights/summaries/stddev/Square#CNN2/weights/summaries/stddev/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
~
"CNN2/weights/summaries/stddev/SqrtSqrt"CNN2/weights/summaries/stddev/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
$CNN2/weights/summaries/stddev_1/tagsConst"/device:GPU:0*0
value'B% BCNN2/weights/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/stddev_1ScalarSummary$CNN2/weights/summaries/stddev_1/tags"CNN2/weights/summaries/stddev/Sqrt"/device:GPU:0*
_output_shapes
: *
T0
n
CNN2/weights/summaries/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
u
$CNN2/weights/summaries/range_1/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
u
$CNN2/weights/summaries/range_1/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/range_1Range$CNN2/weights/summaries/range_1/startCNN2/weights/summaries/Rank_1$CNN2/weights/summaries/range_1/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN2/weights/summaries/MaxMaxCNN2/weights/Variable/readCNN2/weights/summaries/range_1"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN2/weights/summaries/max/tagsConst"/device:GPU:0*+
value"B  BCNN2/weights/summaries/max*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/maxScalarSummaryCNN2/weights/summaries/max/tagsCNN2/weights/summaries/Max"/device:GPU:0*
T0*
_output_shapes
: 
n
CNN2/weights/summaries/Rank_2Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
u
$CNN2/weights/summaries/range_2/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
u
$CNN2/weights/summaries/range_2/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/range_2Range$CNN2/weights/summaries/range_2/startCNN2/weights/summaries/Rank_2$CNN2/weights/summaries/range_2/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
�
CNN2/weights/summaries/MinMinCNN2/weights/Variable/readCNN2/weights/summaries/range_2"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN2/weights/summaries/min/tagsConst"/device:GPU:0*+
value"B  BCNN2/weights/summaries/min*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/minScalarSummaryCNN2/weights/summaries/min/tagsCNN2/weights/summaries/Min"/device:GPU:0*
T0*
_output_shapes
: 
�
$CNN2/weights/summaries/histogram/tagConst"/device:GPU:0*
_output_shapes
: *1
value(B& B CNN2/weights/summaries/histogram*
dtype0
�
 CNN2/weights/summaries/histogramHistogramSummary$CNN2/weights/summaries/histogram/tagCNN2/weights/Variable/read"/device:GPU:0*
_output_shapes
: *
T0
m
CNN2/biases/ConstConst"/device:GPU:0*
_output_shapes
: *
valueB *���=*
dtype0
�
CNN2/biases/Variable
VariableV2"/device:GPU:0*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
CNN2/biases/Variable/AssignAssignCNN2/biases/VariableCNN2/biases/Const"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN2/biases/Variable*
validate_shape(*
_output_shapes
: 
�
CNN2/biases/Variable/readIdentityCNN2/biases/Variable"/device:GPU:0*
T0*'
_class
loc:@CNN2/biases/Variable*
_output_shapes
: 
k
CNN2/biases/summaries/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
r
!CNN2/biases/summaries/range/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
r
!CNN2/biases/summaries/range/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/rangeRange!CNN2/biases/summaries/range/startCNN2/biases/summaries/Rank!CNN2/biases/summaries/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN2/biases/summaries/MeanMeanCNN2/biases/Variable/readCNN2/biases/summaries/range"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN2/biases/summaries/mean/tagsConst"/device:GPU:0*
_output_shapes
: *+
value"B  BCNN2/biases/summaries/mean*
dtype0
�
CNN2/biases/summaries/meanScalarSummaryCNN2/biases/summaries/mean/tagsCNN2/biases/summaries/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
 CNN2/biases/summaries/stddev/subSubCNN2/biases/Variable/readCNN2/biases/summaries/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
#CNN2/biases/summaries/stddev/SquareSquare CNN2/biases/summaries/stddev/sub"/device:GPU:0*
T0*
_output_shapes
: 
{
"CNN2/biases/summaries/stddev/ConstConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
!CNN2/biases/summaries/stddev/MeanMean#CNN2/biases/summaries/stddev/Square"CNN2/biases/summaries/stddev/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
|
!CNN2/biases/summaries/stddev/SqrtSqrt!CNN2/biases/summaries/stddev/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
#CNN2/biases/summaries/stddev_1/tagsConst"/device:GPU:0*
_output_shapes
: */
value&B$ BCNN2/biases/summaries/stddev_1*
dtype0
�
CNN2/biases/summaries/stddev_1ScalarSummary#CNN2/biases/summaries/stddev_1/tags!CNN2/biases/summaries/stddev/Sqrt"/device:GPU:0*
T0*
_output_shapes
: 
m
CNN2/biases/summaries/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
t
#CNN2/biases/summaries/range_1/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
t
#CNN2/biases/summaries/range_1/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/range_1Range#CNN2/biases/summaries/range_1/startCNN2/biases/summaries/Rank_1#CNN2/biases/summaries/range_1/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN2/biases/summaries/MaxMaxCNN2/biases/Variable/readCNN2/biases/summaries/range_1"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN2/biases/summaries/max/tagsConst"/device:GPU:0**
value!B BCNN2/biases/summaries/max*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/maxScalarSummaryCNN2/biases/summaries/max/tagsCNN2/biases/summaries/Max"/device:GPU:0*
T0*
_output_shapes
: 
m
CNN2/biases/summaries/Rank_2Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
t
#CNN2/biases/summaries/range_2/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
t
#CNN2/biases/summaries/range_2/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/range_2Range#CNN2/biases/summaries/range_2/startCNN2/biases/summaries/Rank_2#CNN2/biases/summaries/range_2/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN2/biases/summaries/MinMinCNN2/biases/Variable/readCNN2/biases/summaries/range_2"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
CNN2/biases/summaries/min/tagsConst"/device:GPU:0**
value!B BCNN2/biases/summaries/min*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/minScalarSummaryCNN2/biases/summaries/min/tagsCNN2/biases/summaries/Min"/device:GPU:0*
T0*
_output_shapes
: 
�
#CNN2/biases/summaries/histogram/tagConst"/device:GPU:0*0
value'B% BCNN2/biases/summaries/histogram*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/histogramHistogramSummary#CNN2/biases/summaries/histogram/tagCNN2/biases/Variable/read"/device:GPU:0*
_output_shapes
: *
T0
�
CNN2/Conv2DConv2D#CNN1/batch_normalization/cond/MergeCNN2/weights/Variable/read"/device:GPU:0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������   *
T0
�
CNN2/addAddCNN2/Conv2DCNN2/biases/Variable/read"/device:GPU:0*
T0*/
_output_shapes
:���������   
d
	CNN2/ReluReluCNN2/add"/device:GPU:0*/
_output_shapes
:���������   *
T0
t
CNN2/activations/tagConst"/device:GPU:0*
_output_shapes
: *!
valueB BCNN2/activations*
dtype0
u
CNN2/activationsHistogramSummaryCNN2/activations/tag	CNN2/Relu"/device:GPU:0*
T0*
_output_shapes
: 
�
,batch_normalization_1/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_1/gamma*
valueB *  �?*
dtype0*
_output_shapes
: 
�
batch_normalization_1/gamma
VariableV2"/device:GPU:0*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@batch_normalization_1/gamma
�
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
 batch_normalization_1/gamma/readIdentitybatch_normalization_1/gamma"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: 
�
,batch_normalization_1/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_1/beta*
valueB *    *
dtype0*
_output_shapes
: 
�
batch_normalization_1/beta
VariableV2"/device:GPU:0*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *-
_class#
!loc:@batch_normalization_1/beta
�
!batch_normalization_1/beta/AssignAssignbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros"/device:GPU:0*
_output_shapes
: *
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(
�
batch_normalization_1/beta/readIdentitybatch_normalization_1/beta"/device:GPU:0*
_output_shapes
: *
T0*-
_class#
!loc:@batch_normalization_1/beta
�
3batch_normalization_1/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
valueB *    *
dtype0*
_output_shapes
: 
�
!batch_normalization_1/moving_mean
VariableV2"/device:GPU:0*
shape: *
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container 
�
(batch_normalization_1/moving_mean/AssignAssign!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros"/device:GPU:0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
&batch_normalization_1/moving_mean/readIdentity!batch_normalization_1/moving_mean"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
�
6batch_normalization_1/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB *  �?*
dtype0*
_output_shapes
: 
�
%batch_normalization_1/moving_variance
VariableV2"/device:GPU:0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones"/device:GPU:0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
*batch_normalization_1/moving_variance/readIdentity%batch_normalization_1/moving_variance"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
�
$CNN2/batch_normalization/cond/SwitchSwitchinput/Placeholderinput/Placeholder"/device:GPU:0*
_output_shapes

::*
T0

�
&CNN2/batch_normalization/cond/switch_tIdentity&CNN2/batch_normalization/cond/Switch:1"/device:GPU:0*
_output_shapes
:*
T0

�
&CNN2/batch_normalization/cond/switch_fIdentity$CNN2/batch_normalization/cond/Switch"/device:GPU:0*
_output_shapes
:*
T0

v
%CNN2/batch_normalization/cond/pred_idIdentityinput/Placeholder"/device:GPU:0*
_output_shapes
:*
T0

�
#CNN2/batch_normalization/cond/ConstConst'^CNN2/batch_normalization/cond/switch_t"/device:GPU:0*
_output_shapes
: *
valueB *
dtype0
�
%CNN2/batch_normalization/cond/Const_1Const'^CNN2/batch_normalization/cond/switch_t"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
3CNN2/batch_normalization/cond/FusedBatchNorm/SwitchSwitch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
_class
loc:@CNN2/Relu*J
_output_shapes8
6:���������   :���������   *
T0
�
5CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1Switch batch_normalization_1/gamma/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_1/gamma
�
5CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
: : *
T0
�
,CNN2/batch_normalization/cond/FusedBatchNormFusedBatchNorm5CNN2/batch_normalization/cond/FusedBatchNorm/Switch:17CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1:17CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2:1#CNN2/batch_normalization/cond/Const%CNN2/batch_normalization/cond/Const_1"/device:GPU:0*
epsilon%o�:*
T0*
data_formatNCHW*G
_output_shapes5
3:���������   : : : : *
is_training(
�
5CNN2/batch_normalization/cond/FusedBatchNorm_1/SwitchSwitch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
_class
loc:@CNN2/Relu*J
_output_shapes8
6:���������   :���������   *
T0
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_1/gamma/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
: : 
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
: : 
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_1/moving_mean/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean* 
_output_shapes
: : 
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_1/moving_variance/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*8
_class.
,*loc:@batch_normalization_1/moving_variance* 
_output_shapes
: : *
T0
�
.CNN2/batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm5CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_27CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*G
_output_shapes5
3:���������   : : : : *
is_training( *
epsilon%o�:*
T0*
data_formatNCHW
�
#CNN2/batch_normalization/cond/MergeMerge.CNN2/batch_normalization/cond/FusedBatchNorm_1,CNN2/batch_normalization/cond/FusedBatchNorm"/device:GPU:0*
N*1
_output_shapes
:���������   : *
T0
�
%CNN2/batch_normalization/cond/Merge_1Merge0CNN2/batch_normalization/cond/FusedBatchNorm_1:1.CNN2/batch_normalization/cond/FusedBatchNorm:1"/device:GPU:0*
T0*
N*
_output_shapes

: : 
�
%CNN2/batch_normalization/cond/Merge_2Merge0CNN2/batch_normalization/cond/FusedBatchNorm_1:2.CNN2/batch_normalization/cond/FusedBatchNorm:2"/device:GPU:0*
_output_shapes

: : *
T0*
N
}
)CNN2/batch_normalization/ExpandDims/inputConst"/device:GPU:0*
_output_shapes
: *
valueB
 *
�#<*
dtype0
x
'CNN2/batch_normalization/ExpandDims/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
#CNN2/batch_normalization/ExpandDims
ExpandDims)CNN2/batch_normalization/ExpandDims/input'CNN2/batch_normalization/ExpandDims/dim"/device:GPU:0*
_output_shapes
:*

Tdim0*
T0

+CNN2/batch_normalization/ExpandDims_1/inputConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
z
)CNN2/batch_normalization/ExpandDims_1/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
%CNN2/batch_normalization/ExpandDims_1
ExpandDims+CNN2/batch_normalization/ExpandDims_1/input)CNN2/batch_normalization/ExpandDims_1/dim"/device:GPU:0*

Tdim0*
T0*
_output_shapes
:

&CNN2/batch_normalization/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
 CNN2/batch_normalization/ReshapeReshapeinput/Placeholder&CNN2/batch_normalization/Reshape/shape"/device:GPU:0*
T0
*
Tshape0*
_output_shapes
:
�
CNN2/batch_normalization/SelectSelect CNN2/batch_normalization/Reshape#CNN2/batch_normalization/ExpandDims%CNN2/batch_normalization/ExpandDims_1"/device:GPU:0*
T0*
_output_shapes
:
�
 CNN2/batch_normalization/SqueezeSqueezeCNN2/batch_normalization/Select"/device:GPU:0*
_output_shapes
: *
squeeze_dims
 *
T0
�
-CNN2/batch_normalization/AssignMovingAvg/readIdentity!batch_normalization_1/moving_mean"/device:GPU:0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: *
T0
�
,CNN2/batch_normalization/AssignMovingAvg/SubSub-CNN2/batch_normalization/AssignMovingAvg/read%CNN2/batch_normalization/cond/Merge_1"/device:GPU:0*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
,CNN2/batch_normalization/AssignMovingAvg/MulMul,CNN2/batch_normalization/AssignMovingAvg/Sub CNN2/batch_normalization/Squeeze"/device:GPU:0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: *
T0
�
(CNN2/batch_normalization/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean,CNN2/batch_normalization/AssignMovingAvg/Mul"/device:GPU:0*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
�
/CNN2/batch_normalization/AssignMovingAvg_1/readIdentity%batch_normalization_1/moving_variance"/device:GPU:0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: *
T0
�
.CNN2/batch_normalization/AssignMovingAvg_1/SubSub/CNN2/batch_normalization/AssignMovingAvg_1/read%CNN2/batch_normalization/cond/Merge_2"/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
.CNN2/batch_normalization/AssignMovingAvg_1/MulMul.CNN2/batch_normalization/AssignMovingAvg_1/Sub CNN2/batch_normalization/Squeeze"/device:GPU:0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: *
T0
�
*CNN2/batch_normalization/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance.CNN2/batch_normalization/AssignMovingAvg_1/Mul"/device:GPU:0*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
r
CNN2/batch_norm/tagConst"/device:GPU:0* 
valueB BCNN2/batch_norm*
dtype0*
_output_shapes
: 
�
CNN2/batch_normHistogramSummaryCNN2/batch_norm/tag#CNN2/batch_normalization/cond/Merge"/device:GPU:0*
T0*
_output_shapes
: 
�
#CNN3/weights/truncated_normal/shapeConst"/device:GPU:0*%
valueB"          @   *
dtype0*
_output_shapes
:
v
"CNN3/weights/truncated_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
x
$CNN3/weights/truncated_normal/stddevConst"/device:GPU:0*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
-CNN3/weights/truncated_normal/TruncatedNormalTruncatedNormal#CNN3/weights/truncated_normal/shape"/device:GPU:0*
T0*
dtype0*&
_output_shapes
: @*
seed2 *

seed 
�
!CNN3/weights/truncated_normal/mulMul-CNN3/weights/truncated_normal/TruncatedNormal$CNN3/weights/truncated_normal/stddev"/device:GPU:0*&
_output_shapes
: @*
T0
�
CNN3/weights/truncated_normalAdd!CNN3/weights/truncated_normal/mul"CNN3/weights/truncated_normal/mean"/device:GPU:0*&
_output_shapes
: @*
T0
�
CNN3/weights/Variable
VariableV2"/device:GPU:0*
dtype0*&
_output_shapes
: @*
	container *
shape: @*
shared_name 
�
CNN3/weights/Variable/AssignAssignCNN3/weights/VariableCNN3/weights/truncated_normal"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN3/weights/Variable*
validate_shape(*&
_output_shapes
: @
�
CNN3/weights/Variable/readIdentityCNN3/weights/Variable"/device:GPU:0*&
_output_shapes
: @*
T0*(
_class
loc:@CNN3/weights/Variable
l
CNN3/weights/summaries/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
s
"CNN3/weights/summaries/range/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
s
"CNN3/weights/summaries/range/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/rangeRange"CNN3/weights/summaries/range/startCNN3/weights/summaries/Rank"CNN3/weights/summaries/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN3/weights/summaries/MeanMeanCNN3/weights/Variable/readCNN3/weights/summaries/range"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
 CNN3/weights/summaries/mean/tagsConst"/device:GPU:0*,
value#B! BCNN3/weights/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/meanScalarSummary CNN3/weights/summaries/mean/tagsCNN3/weights/summaries/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
!CNN3/weights/summaries/stddev/subSubCNN3/weights/Variable/readCNN3/weights/summaries/Mean"/device:GPU:0*
T0*&
_output_shapes
: @
�
$CNN3/weights/summaries/stddev/SquareSquare!CNN3/weights/summaries/stddev/sub"/device:GPU:0*
T0*&
_output_shapes
: @
�
#CNN3/weights/summaries/stddev/ConstConst"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
"CNN3/weights/summaries/stddev/MeanMean$CNN3/weights/summaries/stddev/Square#CNN3/weights/summaries/stddev/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
~
"CNN3/weights/summaries/stddev/SqrtSqrt"CNN3/weights/summaries/stddev/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
$CNN3/weights/summaries/stddev_1/tagsConst"/device:GPU:0*
_output_shapes
: *0
value'B% BCNN3/weights/summaries/stddev_1*
dtype0
�
CNN3/weights/summaries/stddev_1ScalarSummary$CNN3/weights/summaries/stddev_1/tags"CNN3/weights/summaries/stddev/Sqrt"/device:GPU:0*
T0*
_output_shapes
: 
n
CNN3/weights/summaries/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
u
$CNN3/weights/summaries/range_1/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
u
$CNN3/weights/summaries/range_1/deltaConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
CNN3/weights/summaries/range_1Range$CNN3/weights/summaries/range_1/startCNN3/weights/summaries/Rank_1$CNN3/weights/summaries/range_1/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN3/weights/summaries/MaxMaxCNN3/weights/Variable/readCNN3/weights/summaries/range_1"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
CNN3/weights/summaries/max/tagsConst"/device:GPU:0*+
value"B  BCNN3/weights/summaries/max*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/maxScalarSummaryCNN3/weights/summaries/max/tagsCNN3/weights/summaries/Max"/device:GPU:0*
T0*
_output_shapes
: 
n
CNN3/weights/summaries/Rank_2Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
u
$CNN3/weights/summaries/range_2/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
u
$CNN3/weights/summaries/range_2/deltaConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
CNN3/weights/summaries/range_2Range$CNN3/weights/summaries/range_2/startCNN3/weights/summaries/Rank_2$CNN3/weights/summaries/range_2/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN3/weights/summaries/MinMinCNN3/weights/Variable/readCNN3/weights/summaries/range_2"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN3/weights/summaries/min/tagsConst"/device:GPU:0*+
value"B  BCNN3/weights/summaries/min*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/minScalarSummaryCNN3/weights/summaries/min/tagsCNN3/weights/summaries/Min"/device:GPU:0*
T0*
_output_shapes
: 
�
$CNN3/weights/summaries/histogram/tagConst"/device:GPU:0*1
value(B& B CNN3/weights/summaries/histogram*
dtype0*
_output_shapes
: 
�
 CNN3/weights/summaries/histogramHistogramSummary$CNN3/weights/summaries/histogram/tagCNN3/weights/Variable/read"/device:GPU:0*
T0*
_output_shapes
: 
m
CNN3/biases/ConstConst"/device:GPU:0*
valueB@*���=*
dtype0*
_output_shapes
:@
�
CNN3/biases/Variable
VariableV2"/device:GPU:0*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
�
CNN3/biases/Variable/AssignAssignCNN3/biases/VariableCNN3/biases/Const"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
CNN3/biases/Variable/readIdentityCNN3/biases/Variable"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
_output_shapes
:@*
T0
k
CNN3/biases/summaries/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
r
!CNN3/biases/summaries/range/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
r
!CNN3/biases/summaries/range/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/rangeRange!CNN3/biases/summaries/range/startCNN3/biases/summaries/Rank!CNN3/biases/summaries/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN3/biases/summaries/MeanMeanCNN3/biases/Variable/readCNN3/biases/summaries/range"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN3/biases/summaries/mean/tagsConst"/device:GPU:0*+
value"B  BCNN3/biases/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/meanScalarSummaryCNN3/biases/summaries/mean/tagsCNN3/biases/summaries/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
 CNN3/biases/summaries/stddev/subSubCNN3/biases/Variable/readCNN3/biases/summaries/Mean"/device:GPU:0*
_output_shapes
:@*
T0
�
#CNN3/biases/summaries/stddev/SquareSquare CNN3/biases/summaries/stddev/sub"/device:GPU:0*
T0*
_output_shapes
:@
{
"CNN3/biases/summaries/stddev/ConstConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
!CNN3/biases/summaries/stddev/MeanMean#CNN3/biases/summaries/stddev/Square"CNN3/biases/summaries/stddev/Const"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
|
!CNN3/biases/summaries/stddev/SqrtSqrt!CNN3/biases/summaries/stddev/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
#CNN3/biases/summaries/stddev_1/tagsConst"/device:GPU:0*/
value&B$ BCNN3/biases/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/stddev_1ScalarSummary#CNN3/biases/summaries/stddev_1/tags!CNN3/biases/summaries/stddev/Sqrt"/device:GPU:0*
T0*
_output_shapes
: 
m
CNN3/biases/summaries/Rank_1Const"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
t
#CNN3/biases/summaries/range_1/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
t
#CNN3/biases/summaries/range_1/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/range_1Range#CNN3/biases/summaries/range_1/startCNN3/biases/summaries/Rank_1#CNN3/biases/summaries/range_1/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN3/biases/summaries/MaxMaxCNN3/biases/Variable/readCNN3/biases/summaries/range_1"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN3/biases/summaries/max/tagsConst"/device:GPU:0**
value!B BCNN3/biases/summaries/max*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/maxScalarSummaryCNN3/biases/summaries/max/tagsCNN3/biases/summaries/Max"/device:GPU:0*
_output_shapes
: *
T0
m
CNN3/biases/summaries/Rank_2Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
t
#CNN3/biases/summaries/range_2/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
t
#CNN3/biases/summaries/range_2/deltaConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
CNN3/biases/summaries/range_2Range#CNN3/biases/summaries/range_2/startCNN3/biases/summaries/Rank_2#CNN3/biases/summaries/range_2/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN3/biases/summaries/MinMinCNN3/biases/Variable/readCNN3/biases/summaries/range_2"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN3/biases/summaries/min/tagsConst"/device:GPU:0*
_output_shapes
: **
value!B BCNN3/biases/summaries/min*
dtype0
�
CNN3/biases/summaries/minScalarSummaryCNN3/biases/summaries/min/tagsCNN3/biases/summaries/Min"/device:GPU:0*
_output_shapes
: *
T0
�
#CNN3/biases/summaries/histogram/tagConst"/device:GPU:0*
_output_shapes
: *0
value'B% BCNN3/biases/summaries/histogram*
dtype0
�
CNN3/biases/summaries/histogramHistogramSummary#CNN3/biases/summaries/histogram/tagCNN3/biases/Variable/read"/device:GPU:0*
T0*
_output_shapes
: 
�
CNN3/Conv2DConv2D#CNN2/batch_normalization/cond/MergeCNN3/weights/Variable/read"/device:GPU:0*
paddingSAME*/
_output_shapes
:���������@*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
CNN3/addAddCNN3/Conv2DCNN3/biases/Variable/read"/device:GPU:0*
T0*/
_output_shapes
:���������@
d
	CNN3/ReluReluCNN3/add"/device:GPU:0*/
_output_shapes
:���������@*
T0
t
CNN3/activations/tagConst"/device:GPU:0*!
valueB BCNN3/activations*
dtype0*
_output_shapes
: 
u
CNN3/activationsHistogramSummaryCNN3/activations/tag	CNN3/Relu"/device:GPU:0*
T0*
_output_shapes
: 
�
,batch_normalization_2/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization_2/gamma
VariableV2"/device:GPU:0*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:*
dtype0*
_output_shapes
:
�
"batch_normalization_2/gamma/AssignAssignbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones"/device:GPU:0*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes
:
�
 batch_normalization_2/gamma/readIdentitybatch_normalization_2/gamma"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
�
,batch_normalization_2/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization_2/beta
VariableV2"/device:GPU:0*
shape:*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container 
�
!batch_normalization_2/beta/AssignAssignbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
:
�
batch_normalization_2/beta/readIdentitybatch_normalization_2/beta"/device:GPU:0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:*
T0
�
3batch_normalization_2/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_2/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
�
!batch_normalization_2/moving_mean
VariableV2"/device:GPU:0*
shared_name *4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container *
shape:*
dtype0*
_output_shapes
:
�
(batch_normalization_2/moving_mean/AssignAssign!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes
:
�
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
6batch_normalization_2/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_2/moving_variance*
valueB*  �?*
dtype0*
_output_shapes
:
�
%batch_normalization_2/moving_variance
VariableV2"/device:GPU:0*
shared_name *8
_class.
,*loc:@batch_normalization_2/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
:
�
,batch_normalization_2/moving_variance/AssignAssign%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones"/device:GPU:0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
*batch_normalization_2/moving_variance/readIdentity%batch_normalization_2/moving_variance"/device:GPU:0*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
$CNN3/batch_normalization/cond/SwitchSwitchinput/Placeholderinput/Placeholder"/device:GPU:0*
_output_shapes

::*
T0

�
&CNN3/batch_normalization/cond/switch_tIdentity&CNN3/batch_normalization/cond/Switch:1"/device:GPU:0*
_output_shapes
:*
T0

�
&CNN3/batch_normalization/cond/switch_fIdentity$CNN3/batch_normalization/cond/Switch"/device:GPU:0*
T0
*
_output_shapes
:
v
%CNN3/batch_normalization/cond/pred_idIdentityinput/Placeholder"/device:GPU:0*
T0
*
_output_shapes
:
�
#CNN3/batch_normalization/cond/ConstConst'^CNN3/batch_normalization/cond/switch_t"/device:GPU:0*
_output_shapes
: *
valueB *
dtype0
�
%CNN3/batch_normalization/cond/Const_1Const'^CNN3/batch_normalization/cond/switch_t"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
3CNN3/batch_normalization/cond/FusedBatchNorm/SwitchSwitch	CNN3/Relu%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
_class
loc:@CNN3/Relu*J
_output_shapes8
6:���������@:���������@*
T0
�
5CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_2/gamma
�
5CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
::*
T0
�
,CNN3/batch_normalization/cond/FusedBatchNormFusedBatchNorm5CNN3/batch_normalization/cond/FusedBatchNorm/Switch:17CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1:17CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2:1#CNN3/batch_normalization/cond/Const%CNN3/batch_normalization/cond/Const_1"/device:GPU:0*
T0*
data_formatNCHW*G
_output_shapes5
3:���������@::::*
is_training(*
epsilon%o�:
�
5CNN3/batch_normalization/cond/FusedBatchNorm_1/SwitchSwitch	CNN3/Relu%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*J
_output_shapes8
6:���������@:���������@*
T0*
_class
loc:@CNN3/Relu
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_2/gamma* 
_output_shapes
::
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
::
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_2/moving_mean/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean* 
_output_shapes
::
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_2/moving_variance/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance* 
_output_shapes
::
�
.CNN3/batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm5CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_27CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
data_formatNCHW*G
_output_shapes5
3:���������@::::*
is_training( *
epsilon%o�:*
T0
�
#CNN3/batch_normalization/cond/MergeMerge.CNN3/batch_normalization/cond/FusedBatchNorm_1,CNN3/batch_normalization/cond/FusedBatchNorm"/device:GPU:0*
N*1
_output_shapes
:���������@: *
T0
�
%CNN3/batch_normalization/cond/Merge_1Merge0CNN3/batch_normalization/cond/FusedBatchNorm_1:1.CNN3/batch_normalization/cond/FusedBatchNorm:1"/device:GPU:0*
N*
_output_shapes

:: *
T0
�
%CNN3/batch_normalization/cond/Merge_2Merge0CNN3/batch_normalization/cond/FusedBatchNorm_1:2.CNN3/batch_normalization/cond/FusedBatchNorm:2"/device:GPU:0*
T0*
N*
_output_shapes

:: 
}
)CNN3/batch_normalization/ExpandDims/inputConst"/device:GPU:0*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
x
'CNN3/batch_normalization/ExpandDims/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
#CNN3/batch_normalization/ExpandDims
ExpandDims)CNN3/batch_normalization/ExpandDims/input'CNN3/batch_normalization/ExpandDims/dim"/device:GPU:0*

Tdim0*
T0*
_output_shapes
:

+CNN3/batch_normalization/ExpandDims_1/inputConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
z
)CNN3/batch_normalization/ExpandDims_1/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
%CNN3/batch_normalization/ExpandDims_1
ExpandDims+CNN3/batch_normalization/ExpandDims_1/input)CNN3/batch_normalization/ExpandDims_1/dim"/device:GPU:0*
T0*
_output_shapes
:*

Tdim0

&CNN3/batch_normalization/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
 CNN3/batch_normalization/ReshapeReshapeinput/Placeholder&CNN3/batch_normalization/Reshape/shape"/device:GPU:0*
T0
*
Tshape0*
_output_shapes
:
�
CNN3/batch_normalization/SelectSelect CNN3/batch_normalization/Reshape#CNN3/batch_normalization/ExpandDims%CNN3/batch_normalization/ExpandDims_1"/device:GPU:0*
_output_shapes
:*
T0
�
 CNN3/batch_normalization/SqueezeSqueezeCNN3/batch_normalization/Select"/device:GPU:0*
_output_shapes
: *
squeeze_dims
 *
T0
�
-CNN3/batch_normalization/AssignMovingAvg/readIdentity!batch_normalization_2/moving_mean"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
,CNN3/batch_normalization/AssignMovingAvg/SubSub-CNN3/batch_normalization/AssignMovingAvg/read%CNN3/batch_normalization/cond/Merge_1"/device:GPU:0*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
,CNN3/batch_normalization/AssignMovingAvg/MulMul,CNN3/batch_normalization/AssignMovingAvg/Sub CNN3/batch_normalization/Squeeze"/device:GPU:0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:*
T0
�
(CNN3/batch_normalization/AssignMovingAvg	AssignSub!batch_normalization_2/moving_mean,CNN3/batch_normalization/AssignMovingAvg/Mul"/device:GPU:0*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
/CNN3/batch_normalization/AssignMovingAvg_1/readIdentity%batch_normalization_2/moving_variance"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
.CNN3/batch_normalization/AssignMovingAvg_1/SubSub/CNN3/batch_normalization/AssignMovingAvg_1/read%CNN3/batch_normalization/cond/Merge_2"/device:GPU:0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
T0
�
.CNN3/batch_normalization/AssignMovingAvg_1/MulMul.CNN3/batch_normalization/AssignMovingAvg_1/Sub CNN3/batch_normalization/Squeeze"/device:GPU:0*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
*CNN3/batch_normalization/AssignMovingAvg_1	AssignSub%batch_normalization_2/moving_variance.CNN3/batch_normalization/AssignMovingAvg_1/Mul"/device:GPU:0*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
r
CNN3/batch_norm/tagConst"/device:GPU:0* 
valueB BCNN3/batch_norm*
dtype0*
_output_shapes
: 
�
CNN3/batch_normHistogramSummaryCNN3/batch_norm/tag#CNN3/batch_normalization/cond/Merge"/device:GPU:0*
T0*
_output_shapes
: 
z
FC1/truncated_normal/shapeConst"/device:GPU:0*
valueB" @     *
dtype0*
_output_shapes
:
m
FC1/truncated_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
o
FC1/truncated_normal/stddevConst"/device:GPU:0*
_output_shapes
: *
valueB
 *���=*
dtype0
�
$FC1/truncated_normal/TruncatedNormalTruncatedNormalFC1/truncated_normal/shape"/device:GPU:0*!
_output_shapes
:���*
seed2 *

seed *
T0*
dtype0
�
FC1/truncated_normal/mulMul$FC1/truncated_normal/TruncatedNormalFC1/truncated_normal/stddev"/device:GPU:0*!
_output_shapes
:���*
T0
�
FC1/truncated_normalAddFC1/truncated_normal/mulFC1/truncated_normal/mean"/device:GPU:0*!
_output_shapes
:���*
T0
�
FC1/Variable
VariableV2"/device:GPU:0*!
_output_shapes
:���*
	container *
shape:���*
shared_name *
dtype0
�
FC1/Variable/AssignAssignFC1/VariableFC1/truncated_normal"/device:GPU:0*
_class
loc:@FC1/Variable*
validate_shape(*!
_output_shapes
:���*
use_locking(*
T0
�
FC1/Variable/readIdentityFC1/Variable"/device:GPU:0*
T0*
_class
loc:@FC1/Variable*!
_output_shapes
:���
g
	FC1/ConstConst"/device:GPU:0*
valueB�*���=*
dtype0*
_output_shapes	
:�
�
FC1/Variable_1
VariableV2"/device:GPU:0*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
�
FC1/Variable_1/AssignAssignFC1/Variable_1	FC1/Const"/device:GPU:0*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@FC1/Variable_1*
validate_shape(
�
FC1/Variable_1/readIdentityFC1/Variable_1"/device:GPU:0*
T0*!
_class
loc:@FC1/Variable_1*
_output_shapes	
:�
q
FC1/Reshape/shapeConst"/device:GPU:0*
valueB"���� @  *
dtype0*
_output_shapes
:
�
FC1/ReshapeReshape#CNN3/batch_normalization/cond/MergeFC1/Reshape/shape"/device:GPU:0*
Tshape0*)
_output_shapes
:�����������*
T0
�

FC1/MatMulMatMulFC1/ReshapeFC1/Variable/read"/device:GPU:0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
q
FC1/addAdd
FC1/MatMulFC1/Variable_1/read"/device:GPU:0*(
_output_shapes
:����������*
T0
[
FC1/ReluReluFC1/add"/device:GPU:0*
T0*(
_output_shapes
:����������
�
,batch_normalization_3/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_3/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
batch_normalization_3/gamma
VariableV2"/device:GPU:0*
dtype0*
_output_shapes	
:�*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:�
�
"batch_normalization_3/gamma/AssignAssignbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones"/device:GPU:0*
_output_shapes	
:�*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(
�
 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma"/device:GPU:0*
_output_shapes	
:�*
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
,batch_normalization_3/beta/Initializer/zerosConst*
_output_shapes	
:�*-
_class#
!loc:@batch_normalization_3/beta*
valueB�*    *
dtype0
�
batch_normalization_3/beta
VariableV2"/device:GPU:0*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:�
�
!batch_normalization_3/beta/AssignAssignbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros"/device:GPU:0*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(
�
batch_normalization_3/beta/readIdentitybatch_normalization_3/beta"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:�
�
3batch_normalization_3/moving_mean/Initializer/zerosConst*
_output_shapes	
:�*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB�*    *
dtype0
�
!batch_normalization_3/moving_mean
VariableV2"/device:GPU:0*
shared_name *4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros"/device:GPU:0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean"/device:GPU:0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:�*
T0
�
6batch_normalization_3/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
%batch_normalization_3/moving_variance
VariableV2"/device:GPU:0*
_output_shapes	
:�*
shared_name *8
_class.
,*loc:@batch_normalization_3/moving_variance*
	container *
shape:�*
dtype0
�
,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones"/device:GPU:0*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
validate_shape(*
_output_shapes	
:�
�
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance"/device:GPU:0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:�*
T0
�
6FC1/batch_normalization/moments/mean/reduction_indicesConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
$FC1/batch_normalization/moments/meanMeanFC1/Relu6FC1/batch_normalization/moments/mean/reduction_indices"/device:GPU:0*
_output_shapes
:	�*

Tidx0*
	keep_dims(*
T0
�
,FC1/batch_normalization/moments/StopGradientStopGradient$FC1/batch_normalization/moments/mean"/device:GPU:0*
_output_shapes
:	�*
T0
�
1FC1/batch_normalization/moments/SquaredDifferenceSquaredDifferenceFC1/Relu,FC1/batch_normalization/moments/StopGradient"/device:GPU:0*(
_output_shapes
:����������*
T0
�
:FC1/batch_normalization/moments/variance/reduction_indicesConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
(FC1/batch_normalization/moments/varianceMean1FC1/batch_normalization/moments/SquaredDifference:FC1/batch_normalization/moments/variance/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims(*
T0*
_output_shapes
:	�
�
'FC1/batch_normalization/moments/SqueezeSqueeze$FC1/batch_normalization/moments/mean"/device:GPU:0*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
)FC1/batch_normalization/moments/Squeeze_1Squeeze(FC1/batch_normalization/moments/variance"/device:GPU:0*
squeeze_dims
 *
T0*
_output_shapes	
:�
w
&FC1/batch_normalization/ExpandDims/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
"FC1/batch_normalization/ExpandDims
ExpandDims'FC1/batch_normalization/moments/Squeeze&FC1/batch_normalization/ExpandDims/dim"/device:GPU:0*
T0*
_output_shapes
:	�*

Tdim0
y
(FC1/batch_normalization/ExpandDims_1/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
$FC1/batch_normalization/ExpandDims_1
ExpandDims&batch_normalization_3/moving_mean/read(FC1/batch_normalization/ExpandDims_1/dim"/device:GPU:0*
_output_shapes
:	�*

Tdim0*
T0
~
%FC1/batch_normalization/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
FC1/batch_normalization/ReshapeReshapeinput/Placeholder%FC1/batch_normalization/Reshape/shape"/device:GPU:0*
T0
*
Tshape0*
_output_shapes
:
�
FC1/batch_normalization/SelectSelectFC1/batch_normalization/Reshape"FC1/batch_normalization/ExpandDims$FC1/batch_normalization/ExpandDims_1"/device:GPU:0*
T0*
_output_shapes
:	�
�
FC1/batch_normalization/SqueezeSqueezeFC1/batch_normalization/Select"/device:GPU:0*
_output_shapes	
:�*
squeeze_dims
 *
T0
y
(FC1/batch_normalization/ExpandDims_2/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
$FC1/batch_normalization/ExpandDims_2
ExpandDims)FC1/batch_normalization/moments/Squeeze_1(FC1/batch_normalization/ExpandDims_2/dim"/device:GPU:0*
T0*
_output_shapes
:	�*

Tdim0
y
(FC1/batch_normalization/ExpandDims_3/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
$FC1/batch_normalization/ExpandDims_3
ExpandDims*batch_normalization_3/moving_variance/read(FC1/batch_normalization/ExpandDims_3/dim"/device:GPU:0*
_output_shapes
:	�*

Tdim0*
T0
�
'FC1/batch_normalization/Reshape_1/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
!FC1/batch_normalization/Reshape_1Reshapeinput/Placeholder'FC1/batch_normalization/Reshape_1/shape"/device:GPU:0*
_output_shapes
:*
T0
*
Tshape0
�
 FC1/batch_normalization/Select_1Select!FC1/batch_normalization/Reshape_1$FC1/batch_normalization/ExpandDims_2$FC1/batch_normalization/ExpandDims_3"/device:GPU:0*
_output_shapes
:	�*
T0
�
!FC1/batch_normalization/Squeeze_1Squeeze FC1/batch_normalization/Select_1"/device:GPU:0*
_output_shapes	
:�*
squeeze_dims
 *
T0
~
*FC1/batch_normalization/ExpandDims_4/inputConst"/device:GPU:0*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
y
(FC1/batch_normalization/ExpandDims_4/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
$FC1/batch_normalization/ExpandDims_4
ExpandDims*FC1/batch_normalization/ExpandDims_4/input(FC1/batch_normalization/ExpandDims_4/dim"/device:GPU:0*
T0*
_output_shapes
:*

Tdim0
~
*FC1/batch_normalization/ExpandDims_5/inputConst"/device:GPU:0*
valueB
 *  �?*
dtype0*
_output_shapes
: 
y
(FC1/batch_normalization/ExpandDims_5/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
$FC1/batch_normalization/ExpandDims_5
ExpandDims*FC1/batch_normalization/ExpandDims_5/input(FC1/batch_normalization/ExpandDims_5/dim"/device:GPU:0*

Tdim0*
T0*
_output_shapes
:
�
'FC1/batch_normalization/Reshape_2/shapeConst"/device:GPU:0*
_output_shapes
:*
valueB:*
dtype0
�
!FC1/batch_normalization/Reshape_2Reshapeinput/Placeholder'FC1/batch_normalization/Reshape_2/shape"/device:GPU:0*
T0
*
Tshape0*
_output_shapes
:
�
 FC1/batch_normalization/Select_2Select!FC1/batch_normalization/Reshape_2$FC1/batch_normalization/ExpandDims_4$FC1/batch_normalization/ExpandDims_5"/device:GPU:0*
T0*
_output_shapes
:
�
!FC1/batch_normalization/Squeeze_2Squeeze FC1/batch_normalization/Select_2"/device:GPU:0*
squeeze_dims
 *
T0*
_output_shapes
: 
�
-FC1/batch_normalization/AssignMovingAvg/sub/xConst"/device:GPU:0*
valueB
 *  �?*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
: 
�
+FC1/batch_normalization/AssignMovingAvg/subSub-FC1/batch_normalization/AssignMovingAvg/sub/x!FC1/batch_normalization/Squeeze_2"/device:GPU:0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: *
T0
�
-FC1/batch_normalization/AssignMovingAvg/sub_1Sub&batch_normalization_3/moving_mean/readFC1/batch_normalization/Squeeze"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:�
�
+FC1/batch_normalization/AssignMovingAvg/mulMul-FC1/batch_normalization/AssignMovingAvg/sub_1+FC1/batch_normalization/AssignMovingAvg/sub"/device:GPU:0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:�*
T0
�
'FC1/batch_normalization/AssignMovingAvg	AssignSub!batch_normalization_3/moving_mean+FC1/batch_normalization/AssignMovingAvg/mul"/device:GPU:0*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:�
�
/FC1/batch_normalization/AssignMovingAvg_1/sub/xConst"/device:GPU:0*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
�
-FC1/batch_normalization/AssignMovingAvg_1/subSub/FC1/batch_normalization/AssignMovingAvg_1/sub/x!FC1/batch_normalization/Squeeze_2"/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
/FC1/batch_normalization/AssignMovingAvg_1/sub_1Sub*batch_normalization_3/moving_variance/read!FC1/batch_normalization/Squeeze_1"/device:GPU:0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:�*
T0
�
-FC1/batch_normalization/AssignMovingAvg_1/mulMul/FC1/batch_normalization/AssignMovingAvg_1/sub_1-FC1/batch_normalization/AssignMovingAvg_1/sub"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:�
�
)FC1/batch_normalization/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance-FC1/batch_normalization/AssignMovingAvg_1/mul"/device:GPU:0*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:�
{
'FC1/batch_normalization/batchnorm/add/yConst"/device:GPU:0*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
%FC1/batch_normalization/batchnorm/addAdd!FC1/batch_normalization/Squeeze_1'FC1/batch_normalization/batchnorm/add/y"/device:GPU:0*
T0*
_output_shapes	
:�
�
'FC1/batch_normalization/batchnorm/RsqrtRsqrt%FC1/batch_normalization/batchnorm/add"/device:GPU:0*
_output_shapes	
:�*
T0
�
%FC1/batch_normalization/batchnorm/mulMul'FC1/batch_normalization/batchnorm/Rsqrt batch_normalization_3/gamma/read"/device:GPU:0*
_output_shapes	
:�*
T0
�
'FC1/batch_normalization/batchnorm/mul_1MulFC1/Relu%FC1/batch_normalization/batchnorm/mul"/device:GPU:0*(
_output_shapes
:����������*
T0
�
'FC1/batch_normalization/batchnorm/mul_2MulFC1/batch_normalization/Squeeze%FC1/batch_normalization/batchnorm/mul"/device:GPU:0*
T0*
_output_shapes	
:�
�
%FC1/batch_normalization/batchnorm/subSubbatch_normalization_3/beta/read'FC1/batch_normalization/batchnorm/mul_2"/device:GPU:0*
_output_shapes	
:�*
T0
�
'FC1/batch_normalization/batchnorm/add_1Add'FC1/batch_normalization/batchnorm/mul_1%FC1/batch_normalization/batchnorm/sub"/device:GPU:0*
T0*(
_output_shapes
:����������
h
dropout/dropout_probPlaceholder"/device:GPU:0*
_output_shapes
:*
shape:*
dtype0
~
Readout/truncated_normal/shapeConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
q
Readout/truncated_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
s
Readout/truncated_normal/stddevConst"/device:GPU:0*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
(Readout/truncated_normal/TruncatedNormalTruncatedNormalReadout/truncated_normal/shape"/device:GPU:0*
T0*
dtype0*
_output_shapes
:	�*
seed2 *

seed 
�
Readout/truncated_normal/mulMul(Readout/truncated_normal/TruncatedNormalReadout/truncated_normal/stddev"/device:GPU:0*
_output_shapes
:	�*
T0
�
Readout/truncated_normalAddReadout/truncated_normal/mulReadout/truncated_normal/mean"/device:GPU:0*
T0*
_output_shapes
:	�
�
Readout/Variable
VariableV2"/device:GPU:0*
_output_shapes
:	�*
	container *
shape:	�*
shared_name *
dtype0
�
Readout/Variable/AssignAssignReadout/VariableReadout/truncated_normal"/device:GPU:0*
use_locking(*
T0*#
_class
loc:@Readout/Variable*
validate_shape(*
_output_shapes
:	�
�
Readout/Variable/readIdentityReadout/Variable"/device:GPU:0*
T0*#
_class
loc:@Readout/Variable*
_output_shapes
:	�
i
Readout/ConstConst"/device:GPU:0*
valueB*���=*
dtype0*
_output_shapes
:
�
Readout/Variable_1
VariableV2"/device:GPU:0*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
Readout/Variable_1/AssignAssignReadout/Variable_1Readout/Const"/device:GPU:0*%
_class
loc:@Readout/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
Readout/Variable_1/readIdentityReadout/Variable_1"/device:GPU:0*
_output_shapes
:*
T0*%
_class
loc:@Readout/Variable_1
�
Readout/MatMulMatMul'FC1/batch_normalization/batchnorm/add_1Readout/Variable/read"/device:GPU:0*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
Readout/predictedAddReadout/MatMulReadout/Variable_1/read"/device:GPU:0*
T0*'
_output_shapes
:���������
i
cross_entropy_total/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
y
cross_entropy_total/ShapeShapeReadout/predicted"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
k
cross_entropy_total/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
{
cross_entropy_total/Shape_1ShapeReadout/predicted"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
j
cross_entropy_total/Sub/yConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
cross_entropy_total/SubSubcross_entropy_total/Rank_1cross_entropy_total/Sub/y"/device:GPU:0*
_output_shapes
: *
T0
�
cross_entropy_total/Slice/beginPackcross_entropy_total/Sub"/device:GPU:0*
_output_shapes
:*
T0*

axis *
N
w
cross_entropy_total/Slice/sizeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
cross_entropy_total/SliceSlicecross_entropy_total/Shape_1cross_entropy_total/Slice/begincross_entropy_total/Slice/size"/device:GPU:0*
_output_shapes
:*
Index0*
T0
�
#cross_entropy_total/concat/values_0Const"/device:GPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
p
cross_entropy_total/concat/axisConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
cross_entropy_total/concatConcatV2#cross_entropy_total/concat/values_0cross_entropy_total/Slicecross_entropy_total/concat/axis"/device:GPU:0*
N*
_output_shapes
:*

Tidx0*
T0
�
cross_entropy_total/ReshapeReshapeReadout/predictedcross_entropy_total/concat"/device:GPU:0*0
_output_shapes
:������������������*
T0*
Tshape0
k
cross_entropy_total/Rank_2Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
~
cross_entropy_total/Shape_2Shapeinput/correct_labels"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
l
cross_entropy_total/Sub_1/yConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
cross_entropy_total/Sub_1Subcross_entropy_total/Rank_2cross_entropy_total/Sub_1/y"/device:GPU:0*
_output_shapes
: *
T0
�
!cross_entropy_total/Slice_1/beginPackcross_entropy_total/Sub_1"/device:GPU:0*
T0*

axis *
N*
_output_shapes
:
y
 cross_entropy_total/Slice_1/sizeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
cross_entropy_total/Slice_1Slicecross_entropy_total/Shape_2!cross_entropy_total/Slice_1/begin cross_entropy_total/Slice_1/size"/device:GPU:0*
Index0*
T0*
_output_shapes
:
�
%cross_entropy_total/concat_1/values_0Const"/device:GPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
r
!cross_entropy_total/concat_1/axisConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
cross_entropy_total/concat_1ConcatV2%cross_entropy_total/concat_1/values_0cross_entropy_total/Slice_1!cross_entropy_total/concat_1/axis"/device:GPU:0*

Tidx0*
T0*
N*
_output_shapes
:
�
cross_entropy_total/Reshape_1Reshapeinput/correct_labelscross_entropy_total/concat_1"/device:GPU:0*
Tshape0*0
_output_shapes
:������������������*
T0
�
1cross_entropy_total/SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitscross_entropy_total/Reshapecross_entropy_total/Reshape_1"/device:GPU:0*?
_output_shapes-
+:���������:������������������*
T0
l
cross_entropy_total/Sub_2/yConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
cross_entropy_total/Sub_2Subcross_entropy_total/Rankcross_entropy_total/Sub_2/y"/device:GPU:0*
_output_shapes
: *
T0
z
!cross_entropy_total/Slice_2/beginConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
 cross_entropy_total/Slice_2/sizePackcross_entropy_total/Sub_2"/device:GPU:0*
_output_shapes
:*
T0*

axis *
N
�
cross_entropy_total/Slice_2Slicecross_entropy_total/Shape!cross_entropy_total/Slice_2/begin cross_entropy_total/Slice_2/size"/device:GPU:0*#
_output_shapes
:���������*
Index0*
T0
�
cross_entropy_total/Reshape_2Reshape1cross_entropy_total/SoftmaxCrossEntropyWithLogitscross_entropy_total/Slice_2"/device:GPU:0*#
_output_shapes
:���������*
T0*
Tshape0
r
cross_entropy_total/ConstConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
cross_entropy_total/MeanMeancross_entropy_total/Reshape_2cross_entropy_total/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
train/gradients/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
train/gradients/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *  �?*
dtype0*
_output_shapes
: 
z
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const"/device:GPU:0*
T0*
_output_shapes
: 
�
;train/gradients/cross_entropy_total/Mean_grad/Reshape/shapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:*
dtype0
�
5train/gradients/cross_entropy_total/Mean_grad/ReshapeReshapetrain/gradients/Fill;train/gradients/cross_entropy_total/Mean_grad/Reshape/shape"/device:GPU:0*
Tshape0*
_output_shapes
:*
T0
�
3train/gradients/cross_entropy_total/Mean_grad/ShapeShapecross_entropy_total/Reshape_2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
2train/gradients/cross_entropy_total/Mean_grad/TileTile5train/gradients/cross_entropy_total/Mean_grad/Reshape3train/gradients/cross_entropy_total/Mean_grad/Shape"/device:GPU:0*

Tmultiples0*
T0*#
_output_shapes
:���������
�
5train/gradients/cross_entropy_total/Mean_grad/Shape_1Shapecross_entropy_total/Reshape_2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
5train/gradients/cross_entropy_total/Mean_grad/Shape_2Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB *
dtype0
�
3train/gradients/cross_entropy_total/Mean_grad/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
2train/gradients/cross_entropy_total/Mean_grad/ProdProd5train/gradients/cross_entropy_total/Mean_grad/Shape_13train/gradients/cross_entropy_total/Mean_grad/Const"/device:GPU:0*
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
5train/gradients/cross_entropy_total/Mean_grad/Const_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB: *H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
dtype0
�
4train/gradients/cross_entropy_total/Mean_grad/Prod_1Prod5train/gradients/cross_entropy_total/Mean_grad/Shape_25train/gradients/cross_entropy_total/Mean_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
_output_shapes
: 
�
7train/gradients/cross_entropy_total/Mean_grad/Maximum/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
value	B :*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
dtype0
�
5train/gradients/cross_entropy_total/Mean_grad/MaximumMaximum4train/gradients/cross_entropy_total/Mean_grad/Prod_17train/gradients/cross_entropy_total/Mean_grad/Maximum/y"/device:GPU:0*
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
_output_shapes
: 
�
6train/gradients/cross_entropy_total/Mean_grad/floordivFloorDiv2train/gradients/cross_entropy_total/Mean_grad/Prod5train/gradients/cross_entropy_total/Mean_grad/Maximum"/device:GPU:0*
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
_output_shapes
: 
�
2train/gradients/cross_entropy_total/Mean_grad/CastCast6train/gradients/cross_entropy_total/Mean_grad/floordiv"/device:GPU:0*
_output_shapes
: *

DstT0*

SrcT0
�
5train/gradients/cross_entropy_total/Mean_grad/truedivRealDiv2train/gradients/cross_entropy_total/Mean_grad/Tile2train/gradients/cross_entropy_total/Mean_grad/Cast"/device:GPU:0*
T0*#
_output_shapes
:���������
�
8train/gradients/cross_entropy_total/Reshape_2_grad/ShapeShape1cross_entropy_total/SoftmaxCrossEntropyWithLogits)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
:train/gradients/cross_entropy_total/Reshape_2_grad/ReshapeReshape5train/gradients/cross_entropy_total/Mean_grad/truediv8train/gradients/cross_entropy_total/Reshape_2_grad/Shape"/device:GPU:0*#
_output_shapes
:���������*
T0*
Tshape0
�
train/gradients/zeros_like	ZerosLike3cross_entropy_total/SoftmaxCrossEntropyWithLogits:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*0
_output_shapes
:������������������*
T0
�
Utrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Qtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims:train/gradients/cross_entropy_total/Reshape_2_grad/ReshapeUtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim"/device:GPU:0*
T0*'
_output_shapes
:���������*

Tdim0
�
Jtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/mulMulQtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/ExpandDims3cross_entropy_total/SoftmaxCrossEntropyWithLogits:1"/device:GPU:0*
T0*0
_output_shapes
:������������������
�
6train/gradients/cross_entropy_total/Reshape_grad/ShapeShapeReadout/predicted)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
8train/gradients/cross_entropy_total/Reshape_grad/ReshapeReshapeJtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/mul6train/gradients/cross_entropy_total/Reshape_grad/Shape"/device:GPU:0*'
_output_shapes
:���������*
T0*
Tshape0
�
,train/gradients/Readout/predicted_grad/ShapeShapeReadout/MatMul)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
.train/gradients/Readout/predicted_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
<train/gradients/Readout/predicted_grad/BroadcastGradientArgsBroadcastGradientArgs,train/gradients/Readout/predicted_grad/Shape.train/gradients/Readout/predicted_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
*train/gradients/Readout/predicted_grad/SumSum8train/gradients/cross_entropy_total/Reshape_grad/Reshape<train/gradients/Readout/predicted_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
.train/gradients/Readout/predicted_grad/ReshapeReshape*train/gradients/Readout/predicted_grad/Sum,train/gradients/Readout/predicted_grad/Shape"/device:GPU:0*
T0*
Tshape0*'
_output_shapes
:���������
�
,train/gradients/Readout/predicted_grad/Sum_1Sum8train/gradients/cross_entropy_total/Reshape_grad/Reshape>train/gradients/Readout/predicted_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
0train/gradients/Readout/predicted_grad/Reshape_1Reshape,train/gradients/Readout/predicted_grad/Sum_1.train/gradients/Readout/predicted_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:
�
7train/gradients/Readout/predicted_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1/^train/gradients/Readout/predicted_grad/Reshape1^train/gradients/Readout/predicted_grad/Reshape_1"/device:GPU:0
�
?train/gradients/Readout/predicted_grad/tuple/control_dependencyIdentity.train/gradients/Readout/predicted_grad/Reshape8^train/gradients/Readout/predicted_grad/tuple/group_deps"/device:GPU:0*
T0*A
_class7
53loc:@train/gradients/Readout/predicted_grad/Reshape*'
_output_shapes
:���������
�
Atrain/gradients/Readout/predicted_grad/tuple/control_dependency_1Identity0train/gradients/Readout/predicted_grad/Reshape_18^train/gradients/Readout/predicted_grad/tuple/group_deps"/device:GPU:0*
T0*C
_class9
75loc:@train/gradients/Readout/predicted_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/Readout/MatMul_grad/MatMulMatMul?train/gradients/Readout/predicted_grad/tuple/control_dependencyReadout/Variable/read"/device:GPU:0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/Readout/MatMul_grad/MatMul_1MatMul'FC1/batch_normalization/batchnorm/add_1?train/gradients/Readout/predicted_grad/tuple/control_dependency"/device:GPU:0*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
�
4train/gradients/Readout/MatMul_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1+^train/gradients/Readout/MatMul_grad/MatMul-^train/gradients/Readout/MatMul_grad/MatMul_1"/device:GPU:0
�
<train/gradients/Readout/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/Readout/MatMul_grad/MatMul5^train/gradients/Readout/MatMul_grad/tuple/group_deps"/device:GPU:0*
T0*=
_class3
1/loc:@train/gradients/Readout/MatMul_grad/MatMul*(
_output_shapes
:����������
�
>train/gradients/Readout/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/Readout/MatMul_grad/MatMul_15^train/gradients/Readout/MatMul_grad/tuple/group_deps"/device:GPU:0*
T0*?
_class5
31loc:@train/gradients/Readout/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
Btrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ShapeShape'FC1/batch_normalization/batchnorm/mul_1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Rtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsBtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ShapeDtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/SumSum<train/gradients/Readout/MatMul_grad/tuple/control_dependencyRtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ReshapeReshape@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/SumBtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape"/device:GPU:0*
T0*
Tshape0*(
_output_shapes
:����������
�
Btrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Sum_1Sum<train/gradients/Readout/MatMul_grad/tuple/control_dependencyTtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ftrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeBtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Sum_1Dtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape_1"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
Mtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1E^train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ReshapeG^train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape_1"/device:GPU:0
�
Utrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityDtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ReshapeN^train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/group_deps"/device:GPU:0*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������
�
Wtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityFtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape_1N^train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/group_deps"/device:GPU:0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ShapeShapeFC1/Relu)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:�*
dtype0
�
Rtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsBtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ShapeDtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mulMulUtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency%FC1/batch_normalization/batchnorm/mul"/device:GPU:0*
T0*(
_output_shapes
:����������
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/SumSum@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mulRtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ReshapeReshape@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/SumBtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Shape"/device:GPU:0*(
_output_shapes
:����������*
T0*
Tshape0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mul_1MulFC1/ReluUtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency"/device:GPU:0*(
_output_shapes
:����������*
T0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Sum_1SumBtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mul_1Ttrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Ftrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeBtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Sum_1Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Shape_1"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
Mtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1E^train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ReshapeG^train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1"/device:GPU:0
�
Utrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityDtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ReshapeN^train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps"/device:GPU:0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������*
T0
�
Wtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityFtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1N^train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps"/device:GPU:0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Btrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Ptrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/ShapeBtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/SumSumWtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Ptrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgs"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Btrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/ReshapeReshape>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Sum@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Shape"/device:GPU:0*
Tshape0*
_output_shapes	
:�*
T0
�
@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Sum_1SumWtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Rtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/NegNeg@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Sum_1"/device:GPU:0*
T0*
_output_shapes
:
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1Reshape>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/NegBtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
Ktrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/ReshapeE^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1"/device:GPU:0
�
Strain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityBtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/ReshapeL^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/group_deps"/device:GPU:0*
T0*U
_classK
IGloc:@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape*
_output_shapes	
:�
�
Utrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityDtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1L^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/group_deps"/device:GPU:0*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1*
_output_shapes	
:�
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:�*
dtype0
�
Rtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsBtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ShapeDtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/mulMulUtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1%FC1/batch_normalization/batchnorm/mul"/device:GPU:0*
_output_shapes	
:�*
T0
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/SumSum@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/mulRtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ReshapeReshape@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/SumBtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Shape"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/mul_1MulFC1/batch_normalization/SqueezeUtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1"/device:GPU:0*
T0*
_output_shapes	
:�
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Sum_1SumBtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/mul_1Ttrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Ftrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape_1ReshapeBtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Sum_1Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Shape_1"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
Mtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1E^train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ReshapeG^train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape_1"/device:GPU:0
�
Utrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityDtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ReshapeN^train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps"/device:GPU:0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape*
_output_shapes	
:�*
T0
�
Wtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityFtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape_1N^train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps"/device:GPU:0*
T0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape_1*
_output_shapes	
:�
�
:train/gradients/FC1/batch_normalization/Squeeze_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
�
<train/gradients/FC1/batch_normalization/Squeeze_grad/ReshapeReshapeUtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency:train/gradients/FC1/batch_normalization/Squeeze_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:	�
�
train/gradients/AddNAddNWtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1Wtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1"/device:GPU:0*
_output_shapes	
:�*
T0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Ptrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/ShapeBtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
>train/gradients/FC1/batch_normalization/batchnorm/mul_grad/mulMultrain/gradients/AddN batch_normalization_3/gamma/read"/device:GPU:0*
T0*
_output_shapes	
:�
�
>train/gradients/FC1/batch_normalization/batchnorm/mul_grad/SumSum>train/gradients/FC1/batch_normalization/batchnorm/mul_grad/mulPtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/ReshapeReshape>train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Sum@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/mul_1Mul'FC1/batch_normalization/batchnorm/Rsqrttrain/gradients/AddN"/device:GPU:0*
_output_shapes	
:�*
T0
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Sum_1Sum@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/mul_1Rtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1Reshape@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Sum_1Btrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Shape_1"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
Ktrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/FC1/batch_normalization/batchnorm/mul_grad/ReshapeE^train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1"/device:GPU:0
�
Strain/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityBtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/ReshapeL^train/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/group_deps"/device:GPU:0*U
_classK
IGloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape*
_output_shapes	
:�*
T0
�
Utrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityDtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1L^train/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/group_deps"/device:GPU:0*
_output_shapes	
:�*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1
�
>train/gradients/FC1/batch_normalization/Select_grad/zeros_likeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
:train/gradients/FC1/batch_normalization/Select_grad/SelectSelectFC1/batch_normalization/Reshape<train/gradients/FC1/batch_normalization/Squeeze_grad/Reshape>train/gradients/FC1/batch_normalization/Select_grad/zeros_like"/device:GPU:0*
_output_shapes
:	�*
T0
�
<train/gradients/FC1/batch_normalization/Select_grad/Select_1SelectFC1/batch_normalization/Reshape>train/gradients/FC1/batch_normalization/Select_grad/zeros_like<train/gradients/FC1/batch_normalization/Squeeze_grad/Reshape"/device:GPU:0*
T0*
_output_shapes
:	�
�
Dtrain/gradients/FC1/batch_normalization/Select_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1;^train/gradients/FC1/batch_normalization/Select_grad/Select=^train/gradients/FC1/batch_normalization/Select_grad/Select_1"/device:GPU:0
�
Ltrain/gradients/FC1/batch_normalization/Select_grad/tuple/control_dependencyIdentity:train/gradients/FC1/batch_normalization/Select_grad/SelectE^train/gradients/FC1/batch_normalization/Select_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:	�*
T0*M
_classC
A?loc:@train/gradients/FC1/batch_normalization/Select_grad/Select
�
Ntrain/gradients/FC1/batch_normalization/Select_grad/tuple/control_dependency_1Identity<train/gradients/FC1/batch_normalization/Select_grad/Select_1E^train/gradients/FC1/batch_normalization/Select_grad/tuple/group_deps"/device:GPU:0*
T0*O
_classE
CAloc:@train/gradients/FC1/batch_normalization/Select_grad/Select_1*
_output_shapes
:	�
�
Ftrain/gradients/FC1/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad'FC1/batch_normalization/batchnorm/RsqrtStrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/control_dependency"/device:GPU:0*
T0*
_output_shapes	
:�
�
=train/gradients/FC1/batch_normalization/ExpandDims_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
?train/gradients/FC1/batch_normalization/ExpandDims_grad/ReshapeReshapeLtrain/gradients/FC1/batch_normalization/Select_grad/tuple/control_dependency=train/gradients/FC1/batch_normalization/ExpandDims_grad/Shape"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
@train/gradients/FC1/batch_normalization/batchnorm/add_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Btrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB *
dtype0
�
Ptrain/gradients/FC1/batch_normalization/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/FC1/batch_normalization/batchnorm/add_grad/ShapeBtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
>train/gradients/FC1/batch_normalization/batchnorm/add_grad/SumSumFtrain/gradients/FC1/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGradPtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/add_grad/ReshapeReshape>train/gradients/FC1/batch_normalization/batchnorm/add_grad/Sum@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Shape"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Sum_1SumFtrain/gradients/FC1/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGradRtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Reshape_1Reshape@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Sum_1Btrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Shape_1"/device:GPU:0*
Tshape0*
_output_shapes
: *
T0
�
Ktrain/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/FC1/batch_normalization/batchnorm/add_grad/ReshapeE^train/gradients/FC1/batch_normalization/batchnorm/add_grad/Reshape_1"/device:GPU:0
�
Strain/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/control_dependencyIdentityBtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/ReshapeL^train/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/group_deps"/device:GPU:0*
T0*U
_classK
IGloc:@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Reshape*
_output_shapes	
:�
�
Utrain/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Reshape_1L^train/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/group_deps"/device:GPU:0*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Reshape_1*
_output_shapes
: 
�
Btrain/gradients/FC1/batch_normalization/moments/Squeeze_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
�
Dtrain/gradients/FC1/batch_normalization/moments/Squeeze_grad/ReshapeReshape?train/gradients/FC1/batch_normalization/ExpandDims_grad/ReshapeBtrain/gradients/FC1/batch_normalization/moments/Squeeze_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:	�
�
<train/gradients/FC1/batch_normalization/Squeeze_1_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0
�
>train/gradients/FC1/batch_normalization/Squeeze_1_grad/ReshapeReshapeStrain/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/control_dependency<train/gradients/FC1/batch_normalization/Squeeze_1_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:	�
�
@train/gradients/FC1/batch_normalization/Select_1_grad/zeros_likeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
<train/gradients/FC1/batch_normalization/Select_1_grad/SelectSelect!FC1/batch_normalization/Reshape_1>train/gradients/FC1/batch_normalization/Squeeze_1_grad/Reshape@train/gradients/FC1/batch_normalization/Select_1_grad/zeros_like"/device:GPU:0*
_output_shapes
:	�*
T0
�
>train/gradients/FC1/batch_normalization/Select_1_grad/Select_1Select!FC1/batch_normalization/Reshape_1@train/gradients/FC1/batch_normalization/Select_1_grad/zeros_like>train/gradients/FC1/batch_normalization/Squeeze_1_grad/Reshape"/device:GPU:0*
_output_shapes
:	�*
T0
�
Ftrain/gradients/FC1/batch_normalization/Select_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1=^train/gradients/FC1/batch_normalization/Select_1_grad/Select?^train/gradients/FC1/batch_normalization/Select_1_grad/Select_1"/device:GPU:0
�
Ntrain/gradients/FC1/batch_normalization/Select_1_grad/tuple/control_dependencyIdentity<train/gradients/FC1/batch_normalization/Select_1_grad/SelectG^train/gradients/FC1/batch_normalization/Select_1_grad/tuple/group_deps"/device:GPU:0*
T0*O
_classE
CAloc:@train/gradients/FC1/batch_normalization/Select_1_grad/Select*
_output_shapes
:	�
�
Ptrain/gradients/FC1/batch_normalization/Select_1_grad/tuple/control_dependency_1Identity>train/gradients/FC1/batch_normalization/Select_1_grad/Select_1G^train/gradients/FC1/batch_normalization/Select_1_grad/tuple/group_deps"/device:GPU:0*
T0*Q
_classG
ECloc:@train/gradients/FC1/batch_normalization/Select_1_grad/Select_1*
_output_shapes
:	�
�
?train/gradients/FC1/batch_normalization/ExpandDims_2_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:�*
dtype0
�
Atrain/gradients/FC1/batch_normalization/ExpandDims_2_grad/ReshapeReshapeNtrain/gradients/FC1/batch_normalization/Select_1_grad/tuple/control_dependency?train/gradients/FC1/batch_normalization/ExpandDims_2_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
Dtrain/gradients/FC1/batch_normalization/moments/Squeeze_1_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0
�
Ftrain/gradients/FC1/batch_normalization/moments/Squeeze_1_grad/ReshapeReshapeAtrain/gradients/FC1/batch_normalization/ExpandDims_2_grad/ReshapeDtrain/gradients/FC1/batch_normalization/moments/Squeeze_1_grad/Shape"/device:GPU:0*
Tshape0*
_output_shapes
:	�*
T0
�
Ctrain/gradients/FC1/batch_normalization/moments/variance_grad/ShapeShape1FC1/batch_normalization/moments/SquaredDifference)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/SizeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Atrain/gradients/FC1/batch_normalization/moments/variance_grad/addAdd:FC1/batch_normalization/moments/variance/reduction_indicesBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Size"/device:GPU:0*
_output_shapes
:*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape
�
Atrain/gradients/FC1/batch_normalization/moments/variance_grad/modFloorModAtrain/gradients/FC1/batch_normalization/moments/variance_grad/addBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Size"/device:GPU:0*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
_output_shapes
:
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0*
_output_shapes
:
�
Itrain/gradients/FC1/batch_normalization/moments/variance_grad/range/startConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B : *V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Itrain/gradients/FC1/batch_normalization/moments/variance_grad/range/deltaConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Ctrain/gradients/FC1/batch_normalization/moments/variance_grad/rangeRangeItrain/gradients/FC1/batch_normalization/moments/variance_grad/range/startBtrain/gradients/FC1/batch_normalization/moments/variance_grad/SizeItrain/gradients/FC1/batch_normalization/moments/variance_grad/range/delta"/device:GPU:0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
_output_shapes
:*

Tidx0
�
Htrain/gradients/FC1/batch_normalization/moments/variance_grad/Fill/valueConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/FillFillEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_1Htrain/gradients/FC1/batch_normalization/moments/variance_grad/Fill/value"/device:GPU:0*
_output_shapes
:*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape
�
Ktrain/gradients/FC1/batch_normalization/moments/variance_grad/DynamicStitchDynamicStitchCtrain/gradients/FC1/batch_normalization/moments/variance_grad/rangeAtrain/gradients/FC1/batch_normalization/moments/variance_grad/modCtrain/gradients/FC1/batch_normalization/moments/variance_grad/ShapeBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Fill"/device:GPU:0*#
_output_shapes
:���������*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
N
�
Gtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
value	B :*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/MaximumMaximumKtrain/gradients/FC1/batch_normalization/moments/variance_grad/DynamicStitchGtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum/y"/device:GPU:0*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*#
_output_shapes
:���������
�
Ftrain/gradients/FC1/batch_normalization/moments/variance_grad/floordivFloorDivCtrain/gradients/FC1/batch_normalization/moments/variance_grad/ShapeEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum"/device:GPU:0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
_output_shapes
:*
T0
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/ReshapeReshapeFtrain/gradients/FC1/batch_normalization/moments/Squeeze_1_grad/ReshapeKtrain/gradients/FC1/batch_normalization/moments/variance_grad/DynamicStitch"/device:GPU:0*
_output_shapes
:*
T0*
Tshape0
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/TileTileEtrain/gradients/FC1/batch_normalization/moments/variance_grad/ReshapeFtrain/gradients/FC1/batch_normalization/moments/variance_grad/floordiv"/device:GPU:0*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2Shape1FC1/batch_normalization/moments/SquaredDifference)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_3Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
�
Ctrain/gradients/FC1/batch_normalization/moments/variance_grad/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/ProdProdEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2Ctrain/gradients/FC1/batch_normalization/moments/variance_grad/Const"/device:GPU:0*
T0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Const_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
�
Dtrain/gradients/FC1/batch_normalization/moments/variance_grad/Prod_1ProdEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_3Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: 
�
Itrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
value	B :*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
dtype0
�
Gtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1MaximumDtrain/gradients/FC1/batch_normalization/moments/variance_grad/Prod_1Itrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1/y"/device:GPU:0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: *
T0
�
Htrain/gradients/FC1/batch_normalization/moments/variance_grad/floordiv_1FloorDivBtrain/gradients/FC1/batch_normalization/moments/variance_grad/ProdGtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1"/device:GPU:0*
T0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: 
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/CastCastHtrain/gradients/FC1/batch_normalization/moments/variance_grad/floordiv_1"/device:GPU:0*

SrcT0*
_output_shapes
: *

DstT0
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/truedivRealDivBtrain/gradients/FC1/batch_normalization/moments/variance_grad/TileBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Cast"/device:GPU:0*
T0*(
_output_shapes
:����������
�
Ltrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ShapeShapeFC1/Relu)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
Ntrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
�
\train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ShapeNtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
Mtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/scalarConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1F^train/gradients/FC1/batch_normalization/moments/variance_grad/truediv"/device:GPU:0*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mulMulMtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/scalarEtrain/gradients/FC1/batch_normalization/moments/variance_grad/truediv"/device:GPU:0*
T0*(
_output_shapes
:����������
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/subSubFC1/Relu,FC1/batch_normalization/moments/StopGradient)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1F^train/gradients/FC1/batch_normalization/moments/variance_grad/truediv"/device:GPU:0*(
_output_shapes
:����������*
T0
�
Ltrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mul_1MulJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mulJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/sub"/device:GPU:0*
T0*(
_output_shapes
:����������
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/SumSumLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mul_1\train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Ntrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ReshapeReshapeJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/SumLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Shape"/device:GPU:0*
T0*
Tshape0*(
_output_shapes
:����������
�
Ltrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Sum_1SumLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mul_1^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Ptrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Reshape_1ReshapeLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Sum_1Ntrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:	�
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/NegNegPtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Reshape_1"/device:GPU:0*
_output_shapes
:	�*
T0
�
Wtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1O^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ReshapeK^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Neg"/device:GPU:0
�
_train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependencyIdentityNtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ReshapeX^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/group_deps"/device:GPU:0*a
_classW
USloc:@train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Reshape*(
_output_shapes
:����������*
T0
�
atrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/NegX^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/group_deps"/device:GPU:0*
T0*]
_classS
QOloc:@train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Neg*
_output_shapes
:	�
�
?train/gradients/FC1/batch_normalization/moments/mean_grad/ShapeShapeFC1/Relu)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/SizeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
=train/gradients/FC1/batch_normalization/moments/mean_grad/addAdd6FC1/batch_normalization/moments/mean/reduction_indices>train/gradients/FC1/batch_normalization/moments/mean_grad/Size"/device:GPU:0*
_output_shapes
:*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape
�
=train/gradients/FC1/batch_normalization/moments/mean_grad/modFloorMod=train/gradients/FC1/batch_normalization/moments/mean_grad/add>train/gradients/FC1/batch_normalization/moments/mean_grad/Size"/device:GPU:0*
_output_shapes
:*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0*
_output_shapes
:
�
Etrain/gradients/FC1/batch_normalization/moments/mean_grad/range/startConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B : *R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
Etrain/gradients/FC1/batch_normalization/moments/mean_grad/range/deltaConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
?train/gradients/FC1/batch_normalization/moments/mean_grad/rangeRangeEtrain/gradients/FC1/batch_normalization/moments/mean_grad/range/start>train/gradients/FC1/batch_normalization/moments/mean_grad/SizeEtrain/gradients/FC1/batch_normalization/moments/mean_grad/range/delta"/device:GPU:0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
_output_shapes
:*

Tidx0
�
Dtrain/gradients/FC1/batch_normalization/moments/mean_grad/Fill/valueConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
value	B :*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/FillFillAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_1Dtrain/gradients/FC1/batch_normalization/moments/mean_grad/Fill/value"/device:GPU:0*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
_output_shapes
:
�
Gtrain/gradients/FC1/batch_normalization/moments/mean_grad/DynamicStitchDynamicStitch?train/gradients/FC1/batch_normalization/moments/mean_grad/range=train/gradients/FC1/batch_normalization/moments/mean_grad/mod?train/gradients/FC1/batch_normalization/moments/mean_grad/Shape>train/gradients/FC1/batch_normalization/moments/mean_grad/Fill"/device:GPU:0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
N*#
_output_shapes
:���������*
T0
�
Ctrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/MaximumMaximumGtrain/gradients/FC1/batch_normalization/moments/mean_grad/DynamicStitchCtrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum/y"/device:GPU:0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*#
_output_shapes
:���������*
T0
�
Btrain/gradients/FC1/batch_normalization/moments/mean_grad/floordivFloorDiv?train/gradients/FC1/batch_normalization/moments/mean_grad/ShapeAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum"/device:GPU:0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
_output_shapes
:*
T0
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/ReshapeReshapeDtrain/gradients/FC1/batch_normalization/moments/Squeeze_grad/ReshapeGtrain/gradients/FC1/batch_normalization/moments/mean_grad/DynamicStitch"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/TileTileAtrain/gradients/FC1/batch_normalization/moments/mean_grad/ReshapeBtrain/gradients/FC1/batch_normalization/moments/mean_grad/floordiv"/device:GPU:0*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2ShapeFC1/Relu)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_3Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0
�
?train/gradients/FC1/batch_normalization/moments/mean_grad/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/ProdProdAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2?train/gradients/FC1/batch_normalization/moments/mean_grad/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Const_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
�
@train/gradients/FC1/batch_normalization/moments/mean_grad/Prod_1ProdAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_3Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Const_1"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2
�
Etrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum_1/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
: 
�
Ctrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum_1Maximum@train/gradients/FC1/batch_normalization/moments/mean_grad/Prod_1Etrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum_1/y"/device:GPU:0*
T0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
_output_shapes
: 
�
Dtrain/gradients/FC1/batch_normalization/moments/mean_grad/floordiv_1FloorDiv>train/gradients/FC1/batch_normalization/moments/mean_grad/ProdCtrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum_1"/device:GPU:0*
T0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
_output_shapes
: 
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/CastCastDtrain/gradients/FC1/batch_normalization/moments/mean_grad/floordiv_1"/device:GPU:0*
_output_shapes
: *

DstT0*

SrcT0
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/truedivRealDiv>train/gradients/FC1/batch_normalization/moments/mean_grad/Tile>train/gradients/FC1/batch_normalization/moments/mean_grad/Cast"/device:GPU:0*
T0*(
_output_shapes
:����������
�
train/gradients/AddN_1AddNUtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependencyAtrain/gradients/FC1/batch_normalization/moments/mean_grad/truediv"/device:GPU:0*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape*
N*(
_output_shapes
:����������
�
&train/gradients/FC1/Relu_grad/ReluGradReluGradtrain/gradients/AddN_1FC1/Relu"/device:GPU:0*(
_output_shapes
:����������*
T0
�
"train/gradients/FC1/add_grad/ShapeShape
FC1/MatMul)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
$train/gradients/FC1/add_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
2train/gradients/FC1/add_grad/BroadcastGradientArgsBroadcastGradientArgs"train/gradients/FC1/add_grad/Shape$train/gradients/FC1/add_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
 train/gradients/FC1/add_grad/SumSum&train/gradients/FC1/Relu_grad/ReluGrad2train/gradients/FC1/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
$train/gradients/FC1/add_grad/ReshapeReshape train/gradients/FC1/add_grad/Sum"train/gradients/FC1/add_grad/Shape"/device:GPU:0*
T0*
Tshape0*(
_output_shapes
:����������
�
"train/gradients/FC1/add_grad/Sum_1Sum&train/gradients/FC1/Relu_grad/ReluGrad4train/gradients/FC1/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
&train/gradients/FC1/add_grad/Reshape_1Reshape"train/gradients/FC1/add_grad/Sum_1$train/gradients/FC1/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
-train/gradients/FC1/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1%^train/gradients/FC1/add_grad/Reshape'^train/gradients/FC1/add_grad/Reshape_1"/device:GPU:0
�
5train/gradients/FC1/add_grad/tuple/control_dependencyIdentity$train/gradients/FC1/add_grad/Reshape.^train/gradients/FC1/add_grad/tuple/group_deps"/device:GPU:0*7
_class-
+)loc:@train/gradients/FC1/add_grad/Reshape*(
_output_shapes
:����������*
T0
�
7train/gradients/FC1/add_grad/tuple/control_dependency_1Identity&train/gradients/FC1/add_grad/Reshape_1.^train/gradients/FC1/add_grad/tuple/group_deps"/device:GPU:0*9
_class/
-+loc:@train/gradients/FC1/add_grad/Reshape_1*
_output_shapes	
:�*
T0
�
&train/gradients/FC1/MatMul_grad/MatMulMatMul5train/gradients/FC1/add_grad/tuple/control_dependencyFC1/Variable/read"/device:GPU:0*)
_output_shapes
:�����������*
transpose_a( *
transpose_b(*
T0
�
(train/gradients/FC1/MatMul_grad/MatMul_1MatMulFC1/Reshape5train/gradients/FC1/add_grad/tuple/control_dependency"/device:GPU:0*!
_output_shapes
:���*
transpose_a(*
transpose_b( *
T0
�
0train/gradients/FC1/MatMul_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1'^train/gradients/FC1/MatMul_grad/MatMul)^train/gradients/FC1/MatMul_grad/MatMul_1"/device:GPU:0
�
8train/gradients/FC1/MatMul_grad/tuple/control_dependencyIdentity&train/gradients/FC1/MatMul_grad/MatMul1^train/gradients/FC1/MatMul_grad/tuple/group_deps"/device:GPU:0*
T0*9
_class/
-+loc:@train/gradients/FC1/MatMul_grad/MatMul*)
_output_shapes
:�����������
�
:train/gradients/FC1/MatMul_grad/tuple/control_dependency_1Identity(train/gradients/FC1/MatMul_grad/MatMul_11^train/gradients/FC1/MatMul_grad/tuple/group_deps"/device:GPU:0*
T0*;
_class1
/-loc:@train/gradients/FC1/MatMul_grad/MatMul_1*!
_output_shapes
:���
�
&train/gradients/FC1/Reshape_grad/ShapeShape#CNN3/batch_normalization/cond/Merge)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
(train/gradients/FC1/Reshape_grad/ReshapeReshape8train/gradients/FC1/MatMul_grad/tuple/control_dependency&train/gradients/FC1/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0*/
_output_shapes
:���������@
�
Btrain/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_gradSwitch(train/gradients/FC1/Reshape_grad/Reshape%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*;
_class1
/-loc:@train/gradients/FC1/Reshape_grad/Reshape*J
_output_shapes8
6:���������@:���������@
�
Itrain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_grad"/device:GPU:0
�
Qtrain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentityBtrain/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_gradJ^train/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*
T0*;
_class1
/-loc:@train/gradients/FC1/Reshape_grad/Reshape*/
_output_shapes
:���������@
�
Strain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/control_dependency_1IdentityDtrain/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_grad:1J^train/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*
T0*;
_class1
/-loc:@train/gradients/FC1/Reshape_grad/Reshape*/
_output_shapes
:���������@
�
train/gradients/zeros_like_1	ZerosLike0CNN3/batch_normalization/cond/FusedBatchNorm_1:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:
�
train/gradients/zeros_like_2	ZerosLike0CNN3/batch_normalization/cond/FusedBatchNorm_1:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0
�
train/gradients/zeros_like_3	ZerosLike0CNN3/batch_normalization/cond/FusedBatchNorm_1:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0
�
train/gradients/zeros_like_4	ZerosLike0CNN3/batch_normalization/cond/FusedBatchNorm_1:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:
�
Rtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Mtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose	Transpose5CNN3/batch_normalization/cond/FusedBatchNorm_1/SwitchRtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/perm"/device:GPU:0*/
_output_shapes
:���������@*
Tperm0*
T0
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Otrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1	TransposeQtrain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/control_dependencyTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/perm"/device:GPU:0*
T0*/
_output_shapes
:���������@*
Tperm0
�
Vtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradOtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1Mtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
T0*
data_formatNHWC*G
_output_shapes5
3:���������@::::*
is_training( *
epsilon%o�:
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Otrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2	TransposeVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/perm"/device:GPU:0*
T0*/
_output_shapes
:���������@*
Tperm0
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1W^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradP^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2"/device:GPU:0
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityOtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������@*
T0*b
_classX
VTloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2
�
^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityXtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityXtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:*
T0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
train/gradients/zeros_like_5	ZerosLike.CNN3/batch_normalization/cond/FusedBatchNorm:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:
�
train/gradients/zeros_like_6	ZerosLike.CNN3/batch_normalization/cond/FusedBatchNorm:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0
�
train/gradients/zeros_like_7	ZerosLike.CNN3/batch_normalization/cond/FusedBatchNorm:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0
�
train/gradients/zeros_like_8	ZerosLike.CNN3/batch_normalization/cond/FusedBatchNorm:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradStrain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/control_dependency_15CNN3/batch_normalization/cond/FusedBatchNorm/Switch:17CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1:1.CNN3/batch_normalization/cond/FusedBatchNorm:3.CNN3/batch_normalization/cond/FusedBatchNorm:4"/device:GPU:0*
T0*
data_formatNCHW*C
_output_shapes1
/:���������@::: : *
is_training(*
epsilon%o�:
�
Rtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad"/device:GPU:0
�
Ztrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradS^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������@*
T0
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:*
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
: *
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
: *
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
train/gradients/SwitchSwitch	CNN3/Relu%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*J
_output_shapes8
6:���������@:���������@
~
train/gradients/Shape_1Shapetrain/gradients/Switch:1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
train/gradients/zeros/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zerosFilltrain/gradients/Shape_1train/gradients/zeros/Const"/device:GPU:0*/
_output_shapes
:���������@*
T0
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMerge\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencytrain/gradients/zeros"/device:GPU:0*
N*1
_output_shapes
:���������@: *
T0
�
train/gradients/Switch_1Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
::
�
train/gradients/Shape_2Shapetrain/gradients/Switch_1:1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
train/gradients/zeros_1/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_1Filltrain/gradients/Shape_2train/gradients/zeros_1/Const"/device:GPU:0*
T0*
_output_shapes
:
�
Vtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMerge^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1train/gradients/zeros_1"/device:GPU:0*
N*
_output_shapes

:: *
T0
�
train/gradients/Switch_2Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
::*
T0
�
train/gradients/Shape_3Shapetrain/gradients/Switch_2:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_2/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_2Filltrain/gradients/Shape_3train/gradients/zeros_2/Const"/device:GPU:0*
T0*
_output_shapes
:
�
Vtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMerge^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2train/gradients/zeros_2"/device:GPU:0*
_output_shapes

:: *
T0*
N
�
train/gradients/Switch_3Switch	CNN3/Relu%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*J
_output_shapes8
6:���������@:���������@*
T0
~
train/gradients/Shape_4Shapetrain/gradients/Switch_3"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_3/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_3Filltrain/gradients/Shape_4train/gradients/zeros_3/Const"/device:GPU:0*
T0*/
_output_shapes
:���������@
�
Rtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergeZtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencytrain/gradients/zeros_3"/device:GPU:0*1
_output_shapes
:���������@: *
T0*
N
�
train/gradients/Switch_4Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
::*
T0
~
train/gradients/Shape_5Shapetrain/gradients/Switch_4"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
train/gradients/zeros_4/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_4Filltrain/gradients/Shape_5train/gradients/zeros_4/Const"/device:GPU:0*
T0*
_output_shapes
:
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMerge\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1train/gradients/zeros_4"/device:GPU:0*
T0*
N*
_output_shapes

:: 
�
train/gradients/Switch_5Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
::*
T0
~
train/gradients/Shape_6Shapetrain/gradients/Switch_5"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
train/gradients/zeros_5/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_5Filltrain/gradients/Shape_6train/gradients/zeros_5/Const"/device:GPU:0*
_output_shapes
:*
T0
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMerge\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2train/gradients/zeros_5"/device:GPU:0*
T0*
N*
_output_shapes

:: 
�
train/gradients/AddN_2AddNTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradRtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad"/device:GPU:0*/
_output_shapes
:���������@*
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N
�
'train/gradients/CNN3/Relu_grad/ReluGradReluGradtrain/gradients/AddN_2	CNN3/Relu"/device:GPU:0*/
_output_shapes
:���������@*
T0
�
train/gradients/AddN_3AddNVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
train/gradients/AddN_4AddNVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad"/device:GPU:0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:*
T0
�
#train/gradients/CNN3/add_grad/ShapeShapeCNN3/Conv2D)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
%train/gradients/CNN3/add_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:@*
dtype0*
_output_shapes
:
�
3train/gradients/CNN3/add_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/CNN3/add_grad/Shape%train/gradients/CNN3/add_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
!train/gradients/CNN3/add_grad/SumSum'train/gradients/CNN3/Relu_grad/ReluGrad3train/gradients/CNN3/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
%train/gradients/CNN3/add_grad/ReshapeReshape!train/gradients/CNN3/add_grad/Sum#train/gradients/CNN3/add_grad/Shape"/device:GPU:0*
T0*
Tshape0*/
_output_shapes
:���������@
�
#train/gradients/CNN3/add_grad/Sum_1Sum'train/gradients/CNN3/Relu_grad/ReluGrad5train/gradients/CNN3/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
'train/gradients/CNN3/add_grad/Reshape_1Reshape#train/gradients/CNN3/add_grad/Sum_1%train/gradients/CNN3/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:@
�
.train/gradients/CNN3/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1&^train/gradients/CNN3/add_grad/Reshape(^train/gradients/CNN3/add_grad/Reshape_1"/device:GPU:0
�
6train/gradients/CNN3/add_grad/tuple/control_dependencyIdentity%train/gradients/CNN3/add_grad/Reshape/^train/gradients/CNN3/add_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������@*
T0*8
_class.
,*loc:@train/gradients/CNN3/add_grad/Reshape
�
8train/gradients/CNN3/add_grad/tuple/control_dependency_1Identity'train/gradients/CNN3/add_grad/Reshape_1/^train/gradients/CNN3/add_grad/tuple/group_deps"/device:GPU:0*:
_class0
.,loc:@train/gradients/CNN3/add_grad/Reshape_1*
_output_shapes
:@*
T0
�
'train/gradients/CNN3/Conv2D_grad/ShapeNShapeN#CNN2/batch_normalization/cond/MergeCNN3/weights/Variable/read)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
::*
T0*
out_type0*
N
�
4train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'train/gradients/CNN3/Conv2D_grad/ShapeNCNN3/weights/Variable/read6train/gradients/CNN3/add_grad/tuple/control_dependency"/device:GPU:0*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC
�
5train/gradients/CNN3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter#CNN2/batch_normalization/cond/Merge)train/gradients/CNN3/Conv2D_grad/ShapeN:16train/gradients/CNN3/add_grad/tuple/control_dependency"/device:GPU:0*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC
�
1train/gradients/CNN3/Conv2D_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_15^train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput6^train/gradients/CNN3/Conv2D_grad/Conv2DBackpropFilter"/device:GPU:0
�
9train/gradients/CNN3/Conv2D_grad/tuple/control_dependencyIdentity4train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput2^train/gradients/CNN3/Conv2D_grad/tuple/group_deps"/device:GPU:0*G
_class=
;9loc:@train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������   *
T0
�
;train/gradients/CNN3/Conv2D_grad/tuple/control_dependency_1Identity5train/gradients/CNN3/Conv2D_grad/Conv2DBackpropFilter2^train/gradients/CNN3/Conv2D_grad/tuple/group_deps"/device:GPU:0*&
_output_shapes
: @*
T0*H
_class>
<:loc:@train/gradients/CNN3/Conv2D_grad/Conv2DBackpropFilter
�
Btrain/gradients/CNN2/batch_normalization/cond/Merge_grad/cond_gradSwitch9train/gradients/CNN3/Conv2D_grad/tuple/control_dependency%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*G
_class=
;9loc:@train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput*J
_output_shapes8
6:���������   :���������   
�
Itrain/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/CNN2/batch_normalization/cond/Merge_grad/cond_grad"/device:GPU:0
�
Qtrain/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentityBtrain/gradients/CNN2/batch_normalization/cond/Merge_grad/cond_gradJ^train/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*
T0*G
_class=
;9loc:@train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������   
�
Strain/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/control_dependency_1IdentityDtrain/gradients/CNN2/batch_normalization/cond/Merge_grad/cond_grad:1J^train/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������   *
T0*G
_class=
;9loc:@train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput
�
train/gradients/zeros_like_9	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
train/gradients/zeros_like_10	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
T0
�
train/gradients/zeros_like_11	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
train/gradients/zeros_like_12	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
Rtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Mtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose	Transpose5CNN2/batch_normalization/cond/FusedBatchNorm_1/SwitchRtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/perm"/device:GPU:0*
T0*/
_output_shapes
:���������   *
Tperm0
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*%
valueB"             *
dtype0
�
Otrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1	TransposeQtrain/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/control_dependencyTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/perm"/device:GPU:0*/
_output_shapes
:���������   *
Tperm0*
T0
�
Vtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradOtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1Mtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
data_formatNHWC*G
_output_shapes5
3:���������   : : : : *
is_training( *
epsilon%o�:*
T0
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Otrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2	TransposeVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/perm"/device:GPU:0*
T0*/
_output_shapes
:���������   *
Tperm0
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1W^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradP^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2"/device:GPU:0
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityOtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*b
_classX
VTloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2*/
_output_shapes
:���������   
�
^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityXtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
: 
�
^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityXtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
: 
�
train/gradients/zeros_like_13	ZerosLike.CNN2/batch_normalization/cond/FusedBatchNorm:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
T0
�
train/gradients/zeros_like_14	ZerosLike.CNN2/batch_normalization/cond/FusedBatchNorm:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
train/gradients/zeros_like_15	ZerosLike.CNN2/batch_normalization/cond/FusedBatchNorm:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
train/gradients/zeros_like_16	ZerosLike.CNN2/batch_normalization/cond/FusedBatchNorm:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradStrain/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/control_dependency_15CNN2/batch_normalization/cond/FusedBatchNorm/Switch:17CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1:1.CNN2/batch_normalization/cond/FusedBatchNorm:3.CNN2/batch_normalization/cond/FusedBatchNorm:4"/device:GPU:0*
data_formatNCHW*C
_output_shapes1
/:���������   : : : : *
is_training(*
epsilon%o�:*
T0
�
Rtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad"/device:GPU:0
�
Ztrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradS^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������   
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
: *
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
train/gradients/Switch_6Switch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*J
_output_shapes8
6:���������   :���������   *
T0
�
train/gradients/Shape_7Shapetrain/gradients/Switch_6:1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
train/gradients/zeros_6/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_6Filltrain/gradients/Shape_7train/gradients/zeros_6/Const"/device:GPU:0*/
_output_shapes
:���������   *
T0
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMerge\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencytrain/gradients/zeros_6"/device:GPU:0*
N*1
_output_shapes
:���������   : *
T0
�
train/gradients/Switch_7Switch batch_normalization_1/gamma/read%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
: : 
�
train/gradients/Shape_8Shapetrain/gradients/Switch_7:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_7/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_7Filltrain/gradients/Shape_8train/gradients/zeros_7/Const"/device:GPU:0*
T0*
_output_shapes
: 
�
Vtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMerge^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1train/gradients/zeros_7"/device:GPU:0*
N*
_output_shapes

: : *
T0
�
train/gradients/Switch_8Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
: : 
�
train/gradients/Shape_9Shapetrain/gradients/Switch_8:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_8/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_8Filltrain/gradients/Shape_9train/gradients/zeros_8/Const"/device:GPU:0*
_output_shapes
: *
T0
�
Vtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMerge^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2train/gradients/zeros_8"/device:GPU:0*
_output_shapes

: : *
T0*
N
�
train/gradients/Switch_9Switch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*J
_output_shapes8
6:���������   :���������   *
T0

train/gradients/Shape_10Shapetrain/gradients/Switch_9"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_9/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_9Filltrain/gradients/Shape_10train/gradients/zeros_9/Const"/device:GPU:0*
T0*/
_output_shapes
:���������   
�
Rtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergeZtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencytrain/gradients/zeros_9"/device:GPU:0*
T0*
N*1
_output_shapes
:���������   : 
�
train/gradients/Switch_10Switch batch_normalization_1/gamma/read%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
: : *
T0
�
train/gradients/Shape_11Shapetrain/gradients/Switch_10"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_10/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_10Filltrain/gradients/Shape_11train/gradients/zeros_10/Const"/device:GPU:0*
_output_shapes
: *
T0
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMerge\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1train/gradients/zeros_10"/device:GPU:0*
_output_shapes

: : *
T0*
N
�
train/gradients/Switch_11Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
: : 
�
train/gradients/Shape_12Shapetrain/gradients/Switch_11"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
train/gradients/zeros_11/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_11Filltrain/gradients/Shape_12train/gradients/zeros_11/Const"/device:GPU:0*
T0*
_output_shapes
: 
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMerge\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2train/gradients/zeros_11"/device:GPU:0*
N*
_output_shapes

: : *
T0
�
train/gradients/AddN_5AddNTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradRtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad"/device:GPU:0*/
_output_shapes
:���������   *
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N
�
'train/gradients/CNN2/Relu_grad/ReluGradReluGradtrain/gradients/AddN_5	CNN2/Relu"/device:GPU:0*/
_output_shapes
:���������   *
T0
�
train/gradients/AddN_6AddNVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
: 
�
train/gradients/AddN_7AddNVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad"/device:GPU:0*
_output_shapes
: *
T0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N
�
#train/gradients/CNN2/add_grad/ShapeShapeCNN2/Conv2D)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
%train/gradients/CNN2/add_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB: *
dtype0
�
3train/gradients/CNN2/add_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/CNN2/add_grad/Shape%train/gradients/CNN2/add_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
!train/gradients/CNN2/add_grad/SumSum'train/gradients/CNN2/Relu_grad/ReluGrad3train/gradients/CNN2/add_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
%train/gradients/CNN2/add_grad/ReshapeReshape!train/gradients/CNN2/add_grad/Sum#train/gradients/CNN2/add_grad/Shape"/device:GPU:0*
T0*
Tshape0*/
_output_shapes
:���������   
�
#train/gradients/CNN2/add_grad/Sum_1Sum'train/gradients/CNN2/Relu_grad/ReluGrad5train/gradients/CNN2/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
'train/gradients/CNN2/add_grad/Reshape_1Reshape#train/gradients/CNN2/add_grad/Sum_1%train/gradients/CNN2/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes
: 
�
.train/gradients/CNN2/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1&^train/gradients/CNN2/add_grad/Reshape(^train/gradients/CNN2/add_grad/Reshape_1"/device:GPU:0
�
6train/gradients/CNN2/add_grad/tuple/control_dependencyIdentity%train/gradients/CNN2/add_grad/Reshape/^train/gradients/CNN2/add_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������   *
T0*8
_class.
,*loc:@train/gradients/CNN2/add_grad/Reshape
�
8train/gradients/CNN2/add_grad/tuple/control_dependency_1Identity'train/gradients/CNN2/add_grad/Reshape_1/^train/gradients/CNN2/add_grad/tuple/group_deps"/device:GPU:0*
T0*:
_class0
.,loc:@train/gradients/CNN2/add_grad/Reshape_1*
_output_shapes
: 
�
'train/gradients/CNN2/Conv2D_grad/ShapeNShapeN#CNN1/batch_normalization/cond/MergeCNN2/weights/Variable/read)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
N* 
_output_shapes
::*
T0
�
4train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'train/gradients/CNN2/Conv2D_grad/ShapeNCNN2/weights/Variable/read6train/gradients/CNN2/add_grad/tuple/control_dependency"/device:GPU:0*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
5train/gradients/CNN2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter#CNN1/batch_normalization/cond/Merge)train/gradients/CNN2/Conv2D_grad/ShapeN:16train/gradients/CNN2/add_grad/tuple/control_dependency"/device:GPU:0*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
1train/gradients/CNN2/Conv2D_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_15^train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput6^train/gradients/CNN2/Conv2D_grad/Conv2DBackpropFilter"/device:GPU:0
�
9train/gradients/CNN2/Conv2D_grad/tuple/control_dependencyIdentity4train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput2^train/gradients/CNN2/Conv2D_grad/tuple/group_deps"/device:GPU:0*
T0*G
_class=
;9loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������@@
�
;train/gradients/CNN2/Conv2D_grad/tuple/control_dependency_1Identity5train/gradients/CNN2/Conv2D_grad/Conv2DBackpropFilter2^train/gradients/CNN2/Conv2D_grad/tuple/group_deps"/device:GPU:0*H
_class>
<:loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: *
T0
�
Btrain/gradients/CNN1/batch_normalization/cond/Merge_grad/cond_gradSwitch9train/gradients/CNN2/Conv2D_grad/tuple/control_dependency%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*J
_output_shapes8
6:���������@@:���������@@*
T0*G
_class=
;9loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput
�
Itrain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/CNN1/batch_normalization/cond/Merge_grad/cond_grad"/device:GPU:0
�
Qtrain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentityBtrain/gradients/CNN1/batch_normalization/cond/Merge_grad/cond_gradJ^train/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*
T0*G
_class=
;9loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������@@
�
Strain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/control_dependency_1IdentityDtrain/gradients/CNN1/batch_normalization/cond/Merge_grad/cond_grad:1J^train/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������@@*
T0*G
_class=
;9loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput
�
train/gradients/zeros_like_17	ZerosLike0CNN1/batch_normalization/cond/FusedBatchNorm_1:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:@*
T0
�
train/gradients/zeros_like_18	ZerosLike0CNN1/batch_normalization/cond/FusedBatchNorm_1:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:@
�
train/gradients/zeros_like_19	ZerosLike0CNN1/batch_normalization/cond/FusedBatchNorm_1:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:@*
T0
�
train/gradients/zeros_like_20	ZerosLike0CNN1/batch_normalization/cond/FusedBatchNorm_1:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:@*
T0
�
Rtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*%
valueB"             *
dtype0
�
Mtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose	Transpose5CNN1/batch_normalization/cond/FusedBatchNorm_1/SwitchRtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/perm"/device:GPU:0*/
_output_shapes
:���������@@*
Tperm0*
T0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*%
valueB"             *
dtype0
�
Otrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1	TransposeQtrain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/control_dependencyTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/perm"/device:GPU:0*/
_output_shapes
:���������@@*
Tperm0*
T0
�
Vtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradOtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1Mtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
T0*
data_formatNHWC*G
_output_shapes5
3:���������@@:@:@:@:@*
is_training( *
epsilon%o�:
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Otrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2	TransposeVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/perm"/device:GPU:0*
T0*/
_output_shapes
:���������@@*
Tperm0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1W^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradP^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2"/device:GPU:0
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityOtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2U^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*b
_classX
VTloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2*/
_output_shapes
:���������@@*
T0
�
^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityXtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1U^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:@
�
^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityXtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2U^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:@*
T0*i
_class_
][loc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
�
train/gradients/zeros_like_21	ZerosLike.CNN1/batch_normalization/cond/FusedBatchNorm:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:@*
T0
�
train/gradients/zeros_like_22	ZerosLike.CNN1/batch_normalization/cond/FusedBatchNorm:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:@
�
train/gradients/zeros_like_23	ZerosLike.CNN1/batch_normalization/cond/FusedBatchNorm:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:@
�
train/gradients/zeros_like_24	ZerosLike.CNN1/batch_normalization/cond/FusedBatchNorm:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:@*
T0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradStrain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/control_dependency_15CNN1/batch_normalization/cond/FusedBatchNorm/Switch:17CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1:1.CNN1/batch_normalization/cond/FusedBatchNorm:3.CNN1/batch_normalization/cond/FusedBatchNorm:4"/device:GPU:0*
T0*
data_formatNCHW*C
_output_shapes1
/:���������@@:@:@: : *
is_training(*
epsilon%o�:
�
Rtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1U^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad"/device:GPU:0
�
Ztrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradS^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������@@
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@*
T0
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
: *
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
train/gradients/Switch_12Switch	CNN1/Relu%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*J
_output_shapes8
6:���������@@:���������@@
�
train/gradients/Shape_13Shapetrain/gradients/Switch_12:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_12/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_12Filltrain/gradients/Shape_13train/gradients/zeros_12/Const"/device:GPU:0*/
_output_shapes
:���������@@*
T0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMerge\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencytrain/gradients/zeros_12"/device:GPU:0*
T0*
N*1
_output_shapes
:���������@@: 
�
train/gradients/Switch_13Switchbatch_normalization/gamma/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
:@:@*
T0
�
train/gradients/Shape_14Shapetrain/gradients/Switch_13:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_13/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_13Filltrain/gradients/Shape_14train/gradients/zeros_13/Const"/device:GPU:0*
T0*
_output_shapes
:@
�
Vtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMerge^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1train/gradients/zeros_13"/device:GPU:0*
T0*
N*
_output_shapes

:@: 
�
train/gradients/Switch_14Switchbatch_normalization/beta/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
:@:@*
T0
�
train/gradients/Shape_15Shapetrain/gradients/Switch_14:1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
train/gradients/zeros_14/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_14Filltrain/gradients/Shape_15train/gradients/zeros_14/Const"/device:GPU:0*
T0*
_output_shapes
:@
�
Vtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMerge^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2train/gradients/zeros_14"/device:GPU:0*
N*
_output_shapes

:@: *
T0
�
train/gradients/Switch_15Switch	CNN1/Relu%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*J
_output_shapes8
6:���������@@:���������@@*
T0
�
train/gradients/Shape_16Shapetrain/gradients/Switch_15"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
train/gradients/zeros_15/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_15Filltrain/gradients/Shape_16train/gradients/zeros_15/Const"/device:GPU:0*/
_output_shapes
:���������@@*
T0
�
Rtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergeZtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencytrain/gradients/zeros_15"/device:GPU:0*
T0*
N*1
_output_shapes
:���������@@: 
�
train/gradients/Switch_16Switchbatch_normalization/gamma/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
:@:@
�
train/gradients/Shape_17Shapetrain/gradients/Switch_16"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
train/gradients/zeros_16/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_16Filltrain/gradients/Shape_17train/gradients/zeros_16/Const"/device:GPU:0*
_output_shapes
:@*
T0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMerge\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1train/gradients/zeros_16"/device:GPU:0*
T0*
N*
_output_shapes

:@: 
�
train/gradients/Switch_17Switchbatch_normalization/beta/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
:@:@
�
train/gradients/Shape_18Shapetrain/gradients/Switch_17"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
train/gradients/zeros_17/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_17Filltrain/gradients/Shape_18train/gradients/zeros_17/Const"/device:GPU:0*
_output_shapes
:@*
T0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMerge\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2train/gradients/zeros_17"/device:GPU:0*
T0*
N*
_output_shapes

:@: 
�
train/gradients/AddN_8AddNTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradRtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������@@
�
'train/gradients/CNN1/Relu_grad/ReluGradReluGradtrain/gradients/AddN_8	CNN1/Relu"/device:GPU:0*/
_output_shapes
:���������@@*
T0
�
train/gradients/AddN_9AddNVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad"/device:GPU:0*
_output_shapes
:@*
T0*i
_class_
][loc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N
�
train/gradients/AddN_10AddNVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:@
�
#train/gradients/CNN1/add_grad/ShapeShapeCNN1/Conv2D)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
%train/gradients/CNN1/add_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:*
dtype0
�
3train/gradients/CNN1/add_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/CNN1/add_grad/Shape%train/gradients/CNN1/add_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
!train/gradients/CNN1/add_grad/SumSum'train/gradients/CNN1/Relu_grad/ReluGrad3train/gradients/CNN1/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
%train/gradients/CNN1/add_grad/ReshapeReshape!train/gradients/CNN1/add_grad/Sum#train/gradients/CNN1/add_grad/Shape"/device:GPU:0*
Tshape0*/
_output_shapes
:���������@@*
T0
�
#train/gradients/CNN1/add_grad/Sum_1Sum'train/gradients/CNN1/Relu_grad/ReluGrad5train/gradients/CNN1/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
'train/gradients/CNN1/add_grad/Reshape_1Reshape#train/gradients/CNN1/add_grad/Sum_1%train/gradients/CNN1/add_grad/Shape_1"/device:GPU:0*
Tshape0*
_output_shapes
:*
T0
�
.train/gradients/CNN1/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1&^train/gradients/CNN1/add_grad/Reshape(^train/gradients/CNN1/add_grad/Reshape_1"/device:GPU:0
�
6train/gradients/CNN1/add_grad/tuple/control_dependencyIdentity%train/gradients/CNN1/add_grad/Reshape/^train/gradients/CNN1/add_grad/tuple/group_deps"/device:GPU:0*
T0*8
_class.
,*loc:@train/gradients/CNN1/add_grad/Reshape*/
_output_shapes
:���������@@
�
8train/gradients/CNN1/add_grad/tuple/control_dependency_1Identity'train/gradients/CNN1/add_grad/Reshape_1/^train/gradients/CNN1/add_grad/tuple/group_deps"/device:GPU:0*:
_class0
.,loc:@train/gradients/CNN1/add_grad/Reshape_1*
_output_shapes
:*
T0
�
'train/gradients/CNN1/Conv2D_grad/ShapeNShapeNinput/imagesCNN1/weights/Variable/read)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
N* 
_output_shapes
::
�
4train/gradients/CNN1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'train/gradients/CNN1/Conv2D_grad/ShapeNCNN1/weights/Variable/read6train/gradients/CNN1/add_grad/tuple/control_dependency"/device:GPU:0*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC
�
5train/gradients/CNN1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput/images)train/gradients/CNN1/Conv2D_grad/ShapeN:16train/gradients/CNN1/add_grad/tuple/control_dependency"/device:GPU:0*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
�
1train/gradients/CNN1/Conv2D_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_15^train/gradients/CNN1/Conv2D_grad/Conv2DBackpropInput6^train/gradients/CNN1/Conv2D_grad/Conv2DBackpropFilter"/device:GPU:0
�
9train/gradients/CNN1/Conv2D_grad/tuple/control_dependencyIdentity4train/gradients/CNN1/Conv2D_grad/Conv2DBackpropInput2^train/gradients/CNN1/Conv2D_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������@@*
T0*G
_class=
;9loc:@train/gradients/CNN1/Conv2D_grad/Conv2DBackpropInput
�
;train/gradients/CNN1/Conv2D_grad/tuple/control_dependency_1Identity5train/gradients/CNN1/Conv2D_grad/Conv2DBackpropFilter2^train/gradients/CNN1/Conv2D_grad/tuple/group_deps"/device:GPU:0*&
_output_shapes
:*
T0*H
_class>
<:loc:@train/gradients/CNN1/Conv2D_grad/Conv2DBackpropFilter
�
train/beta1_power/initial_valueConst"/device:GPU:0*
valueB
 *fff?*'
_class
loc:@CNN1/biases/Variable*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2"/device:GPU:0*
shared_name *'
_class
loc:@CNN1/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
train/beta1_power/readIdentitytrain/beta1_power"/device:GPU:0*
T0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
: 
�
train/beta2_power/initial_valueConst"/device:GPU:0*
_output_shapes
: *
valueB
 *w�?*'
_class
loc:@CNN1/biases/Variable*
dtype0
�
train/beta2_power
VariableV2"/device:GPU:0*
shape: *
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@CNN1/biases/Variable*
	container 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value"/device:GPU:0*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(
�
train/beta2_power/readIdentitytrain/beta2_power"/device:GPU:0*
_output_shapes
: *
T0*'
_class
loc:@CNN1/biases/Variable
�
,CNN1/weights/Variable/Adam/Initializer/zerosConst"/device:GPU:0*&
_output_shapes
:*(
_class
loc:@CNN1/weights/Variable*%
valueB*    *
dtype0
�
CNN1/weights/Variable/Adam
VariableV2"/device:GPU:0*
dtype0*&
_output_shapes
:*
shared_name *(
_class
loc:@CNN1/weights/Variable*
	container *
shape:
�
!CNN1/weights/Variable/Adam/AssignAssignCNN1/weights/Variable/Adam,CNN1/weights/Variable/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN1/weights/Variable*
validate_shape(*&
_output_shapes
:
�
CNN1/weights/Variable/Adam/readIdentityCNN1/weights/Variable/Adam"/device:GPU:0*(
_class
loc:@CNN1/weights/Variable*&
_output_shapes
:*
T0
�
.CNN1/weights/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*&
_output_shapes
:*(
_class
loc:@CNN1/weights/Variable*%
valueB*    *
dtype0
�
CNN1/weights/Variable/Adam_1
VariableV2"/device:GPU:0*&
_output_shapes
:*
shared_name *(
_class
loc:@CNN1/weights/Variable*
	container *
shape:*
dtype0
�
#CNN1/weights/Variable/Adam_1/AssignAssignCNN1/weights/Variable/Adam_1.CNN1/weights/Variable/Adam_1/Initializer/zeros"/device:GPU:0*&
_output_shapes
:*
use_locking(*
T0*(
_class
loc:@CNN1/weights/Variable*
validate_shape(
�
!CNN1/weights/Variable/Adam_1/readIdentityCNN1/weights/Variable/Adam_1"/device:GPU:0*&
_output_shapes
:*
T0*(
_class
loc:@CNN1/weights/Variable
�
+CNN1/biases/Variable/Adam/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:*'
_class
loc:@CNN1/biases/Variable*
valueB*    *
dtype0
�
CNN1/biases/Variable/Adam
VariableV2"/device:GPU:0*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@CNN1/biases/Variable
�
 CNN1/biases/Variable/Adam/AssignAssignCNN1/biases/Variable/Adam+CNN1/biases/Variable/Adam/Initializer/zeros"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(
�
CNN1/biases/Variable/Adam/readIdentityCNN1/biases/Variable/Adam"/device:GPU:0*
T0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
:
�
-CNN1/biases/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
valueB*    *
dtype0*
_output_shapes
:
�
CNN1/biases/Variable/Adam_1
VariableV2"/device:GPU:0*
shared_name *'
_class
loc:@CNN1/biases/Variable*
	container *
shape:*
dtype0*
_output_shapes
:
�
"CNN1/biases/Variable/Adam_1/AssignAssignCNN1/biases/Variable/Adam_1-CNN1/biases/Variable/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
:
�
 CNN1/biases/Variable/Adam_1/readIdentityCNN1/biases/Variable/Adam_1"/device:GPU:0*
T0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
:
�
0batch_normalization/gamma/Adam/Initializer/zerosConst"/device:GPU:0*,
_class"
 loc:@batch_normalization/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
�
batch_normalization/gamma/Adam
VariableV2"/device:GPU:0*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
%batch_normalization/gamma/Adam/AssignAssignbatch_normalization/gamma/Adam0batch_normalization/gamma/Adam/Initializer/zeros"/device:GPU:0*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(
�
#batch_normalization/gamma/Adam/readIdentitybatch_normalization/gamma/Adam"/device:GPU:0*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
�
2batch_normalization/gamma/Adam_1/Initializer/zerosConst"/device:GPU:0*,
_class"
 loc:@batch_normalization/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
�
 batch_normalization/gamma/Adam_1
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:@*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:@
�
'batch_normalization/gamma/Adam_1/AssignAssign batch_normalization/gamma/Adam_12batch_normalization/gamma/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:@
�
%batch_normalization/gamma/Adam_1/readIdentity batch_normalization/gamma/Adam_1"/device:GPU:0*
_output_shapes
:@*
T0*,
_class"
 loc:@batch_normalization/gamma
�
/batch_normalization/beta/Adam/Initializer/zerosConst"/device:GPU:0*+
_class!
loc:@batch_normalization/beta*
valueB@*    *
dtype0*
_output_shapes
:@
�
batch_normalization/beta/Adam
VariableV2"/device:GPU:0*+
_class!
loc:@batch_normalization/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
$batch_normalization/beta/Adam/AssignAssignbatch_normalization/beta/Adam/batch_normalization/beta/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:@
�
"batch_normalization/beta/Adam/readIdentitybatch_normalization/beta/Adam"/device:GPU:0*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
�
1batch_normalization/beta/Adam_1/Initializer/zerosConst"/device:GPU:0*+
_class!
loc:@batch_normalization/beta*
valueB@*    *
dtype0*
_output_shapes
:@
�
batch_normalization/beta/Adam_1
VariableV2"/device:GPU:0*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
&batch_normalization/beta/Adam_1/AssignAssignbatch_normalization/beta/Adam_11batch_normalization/beta/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:@
�
$batch_normalization/beta/Adam_1/readIdentitybatch_normalization/beta/Adam_1"/device:GPU:0*
T0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@
�
,CNN2/weights/Variable/Adam/Initializer/zerosConst"/device:GPU:0*(
_class
loc:@CNN2/weights/Variable*%
valueB *    *
dtype0*&
_output_shapes
: 
�
CNN2/weights/Variable/Adam
VariableV2"/device:GPU:0*
dtype0*&
_output_shapes
: *
shared_name *(
_class
loc:@CNN2/weights/Variable*
	container *
shape: 
�
!CNN2/weights/Variable/Adam/AssignAssignCNN2/weights/Variable/Adam,CNN2/weights/Variable/Adam/Initializer/zeros"/device:GPU:0*(
_class
loc:@CNN2/weights/Variable*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0
�
CNN2/weights/Variable/Adam/readIdentityCNN2/weights/Variable/Adam"/device:GPU:0*(
_class
loc:@CNN2/weights/Variable*&
_output_shapes
: *
T0
�
.CNN2/weights/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*(
_class
loc:@CNN2/weights/Variable*%
valueB *    *
dtype0*&
_output_shapes
: 
�
CNN2/weights/Variable/Adam_1
VariableV2"/device:GPU:0*
dtype0*&
_output_shapes
: *
shared_name *(
_class
loc:@CNN2/weights/Variable*
	container *
shape: 
�
#CNN2/weights/Variable/Adam_1/AssignAssignCNN2/weights/Variable/Adam_1.CNN2/weights/Variable/Adam_1/Initializer/zeros"/device:GPU:0*&
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@CNN2/weights/Variable*
validate_shape(
�
!CNN2/weights/Variable/Adam_1/readIdentityCNN2/weights/Variable/Adam_1"/device:GPU:0*&
_output_shapes
: *
T0*(
_class
loc:@CNN2/weights/Variable
�
+CNN2/biases/Variable/Adam/Initializer/zerosConst"/device:GPU:0*'
_class
loc:@CNN2/biases/Variable*
valueB *    *
dtype0*
_output_shapes
: 
�
CNN2/biases/Variable/Adam
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@CNN2/biases/Variable*
	container *
shape: 
�
 CNN2/biases/Variable/Adam/AssignAssignCNN2/biases/Variable/Adam+CNN2/biases/Variable/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN2/biases/Variable*
validate_shape(*
_output_shapes
: 
�
CNN2/biases/Variable/Adam/readIdentityCNN2/biases/Variable/Adam"/device:GPU:0*'
_class
loc:@CNN2/biases/Variable*
_output_shapes
: *
T0
�
-CNN2/biases/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*'
_class
loc:@CNN2/biases/Variable*
valueB *    *
dtype0*
_output_shapes
: 
�
CNN2/biases/Variable/Adam_1
VariableV2"/device:GPU:0*
shared_name *'
_class
loc:@CNN2/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
�
"CNN2/biases/Variable/Adam_1/AssignAssignCNN2/biases/Variable/Adam_1-CNN2/biases/Variable/Adam_1/Initializer/zeros"/device:GPU:0*'
_class
loc:@CNN2/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
 CNN2/biases/Variable/Adam_1/readIdentityCNN2/biases/Variable/Adam_1"/device:GPU:0*
_output_shapes
: *
T0*'
_class
loc:@CNN2/biases/Variable
�
2batch_normalization_1/gamma/Adam/Initializer/zerosConst"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma*
valueB *    *
dtype0*
_output_shapes
: 
�
 batch_normalization_1/gamma/Adam
VariableV2"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
'batch_normalization_1/gamma/Adam/AssignAssign batch_normalization_1/gamma/Adam2batch_normalization_1/gamma/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes
: 
�
%batch_normalization_1/gamma/Adam/readIdentity batch_normalization_1/gamma/Adam"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: 
�
4batch_normalization_1/gamma/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_1/gamma*
valueB *    *
dtype0
�
"batch_normalization_1/gamma/Adam_1
VariableV2"/device:GPU:0*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@batch_normalization_1/gamma
�
)batch_normalization_1/gamma/Adam_1/AssignAssign"batch_normalization_1/gamma/Adam_14batch_normalization_1/gamma/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(
�
'batch_normalization_1/gamma/Adam_1/readIdentity"batch_normalization_1/gamma/Adam_1"/device:GPU:0*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_1/gamma
�
1batch_normalization_1/beta/Adam/Initializer/zerosConst"/device:GPU:0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_1/beta*
valueB *    *
dtype0
�
batch_normalization_1/beta/Adam
VariableV2"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
&batch_normalization_1/beta/Adam/AssignAssignbatch_normalization_1/beta/Adam1batch_normalization_1/beta/Adam/Initializer/zeros"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
$batch_normalization_1/beta/Adam/readIdentitybatch_normalization_1/beta/Adam"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: *
T0
�
3batch_normalization_1/beta/Adam_1/Initializer/zerosConst"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta*
valueB *    *
dtype0*
_output_shapes
: 
�
!batch_normalization_1/beta/Adam_1
VariableV2"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
(batch_normalization_1/beta/Adam_1/AssignAssign!batch_normalization_1/beta/Adam_13batch_normalization_1/beta/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes
: *
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(
�
&batch_normalization_1/beta/Adam_1/readIdentity!batch_normalization_1/beta/Adam_1"/device:GPU:0*
_output_shapes
: *
T0*-
_class#
!loc:@batch_normalization_1/beta
�
,CNN3/weights/Variable/Adam/Initializer/zerosConst"/device:GPU:0*(
_class
loc:@CNN3/weights/Variable*%
valueB @*    *
dtype0*&
_output_shapes
: @
�
CNN3/weights/Variable/Adam
VariableV2"/device:GPU:0*(
_class
loc:@CNN3/weights/Variable*
	container *
shape: @*
dtype0*&
_output_shapes
: @*
shared_name 
�
!CNN3/weights/Variable/Adam/AssignAssignCNN3/weights/Variable/Adam,CNN3/weights/Variable/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN3/weights/Variable*
validate_shape(*&
_output_shapes
: @
�
CNN3/weights/Variable/Adam/readIdentityCNN3/weights/Variable/Adam"/device:GPU:0*
T0*(
_class
loc:@CNN3/weights/Variable*&
_output_shapes
: @
�
.CNN3/weights/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*(
_class
loc:@CNN3/weights/Variable*%
valueB @*    *
dtype0*&
_output_shapes
: @
�
CNN3/weights/Variable/Adam_1
VariableV2"/device:GPU:0*
dtype0*&
_output_shapes
: @*
shared_name *(
_class
loc:@CNN3/weights/Variable*
	container *
shape: @
�
#CNN3/weights/Variable/Adam_1/AssignAssignCNN3/weights/Variable/Adam_1.CNN3/weights/Variable/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN3/weights/Variable*
validate_shape(*&
_output_shapes
: @
�
!CNN3/weights/Variable/Adam_1/readIdentityCNN3/weights/Variable/Adam_1"/device:GPU:0*&
_output_shapes
: @*
T0*(
_class
loc:@CNN3/weights/Variable
�
+CNN3/biases/Variable/Adam/Initializer/zerosConst"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
valueB@*    *
dtype0*
_output_shapes
:@
�
CNN3/biases/Variable/Adam
VariableV2"/device:GPU:0*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@CNN3/biases/Variable*
	container 
�
 CNN3/biases/Variable/Adam/AssignAssignCNN3/biases/Variable/Adam+CNN3/biases/Variable/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN3/biases/Variable*
validate_shape(*
_output_shapes
:@
�
CNN3/biases/Variable/Adam/readIdentityCNN3/biases/Variable/Adam"/device:GPU:0*
T0*'
_class
loc:@CNN3/biases/Variable*
_output_shapes
:@
�
-CNN3/biases/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:@*'
_class
loc:@CNN3/biases/Variable*
valueB@*    *
dtype0
�
CNN3/biases/Variable/Adam_1
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@CNN3/biases/Variable*
	container *
shape:@
�
"CNN3/biases/Variable/Adam_1/AssignAssignCNN3/biases/Variable/Adam_1-CNN3/biases/Variable/Adam_1/Initializer/zeros"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
 CNN3/biases/Variable/Adam_1/readIdentityCNN3/biases/Variable/Adam_1"/device:GPU:0*
_output_shapes
:@*
T0*'
_class
loc:@CNN3/biases/Variable
�
2batch_normalization_2/gamma/Adam/Initializer/zerosConst"/device:GPU:0*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
 batch_normalization_2/gamma/Adam
VariableV2"/device:GPU:0*.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
'batch_normalization_2/gamma/Adam/AssignAssign batch_normalization_2/gamma/Adam2batch_normalization_2/gamma/Adam/Initializer/zeros"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(
�
%batch_normalization_2/gamma/Adam/readIdentity batch_normalization_2/gamma/Adam"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
�
4batch_normalization_2/gamma/Adam_1/Initializer/zerosConst"/device:GPU:0*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
"batch_normalization_2/gamma/Adam_1
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:
�
)batch_normalization_2/gamma/Adam_1/AssignAssign"batch_normalization_2/gamma/Adam_14batch_normalization_2/gamma/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(
�
'batch_normalization_2/gamma/Adam_1/readIdentity"batch_normalization_2/gamma/Adam_1"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
�
1batch_normalization_2/beta/Adam/Initializer/zerosConst"/device:GPU:0*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization_2/beta/Adam
VariableV2"/device:GPU:0*
shape:*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container 
�
&batch_normalization_2/beta/Adam/AssignAssignbatch_normalization_2/beta/Adam1batch_normalization_2/beta/Adam/Initializer/zeros"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(
�
$batch_normalization_2/beta/Adam/readIdentitybatch_normalization_2/beta/Adam"/device:GPU:0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:*
T0
�
3batch_normalization_2/beta/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
dtype0
�
!batch_normalization_2/beta/Adam_1
VariableV2"/device:GPU:0*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:*
dtype0*
_output_shapes
:
�
(batch_normalization_2/beta/Adam_1/AssignAssign!batch_normalization_2/beta/Adam_13batch_normalization_2/beta/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(
�
&batch_normalization_2/beta/Adam_1/readIdentity!batch_normalization_2/beta/Adam_1"/device:GPU:0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:*
T0
�
#FC1/Variable/Adam/Initializer/zerosConst"/device:GPU:0*
_class
loc:@FC1/Variable* 
valueB���*    *
dtype0*!
_output_shapes
:���
�
FC1/Variable/Adam
VariableV2"/device:GPU:0*
dtype0*!
_output_shapes
:���*
shared_name *
_class
loc:@FC1/Variable*
	container *
shape:���
�
FC1/Variable/Adam/AssignAssignFC1/Variable/Adam#FC1/Variable/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*
_class
loc:@FC1/Variable*
validate_shape(*!
_output_shapes
:���
�
FC1/Variable/Adam/readIdentityFC1/Variable/Adam"/device:GPU:0*!
_output_shapes
:���*
T0*
_class
loc:@FC1/Variable
�
%FC1/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*
_class
loc:@FC1/Variable* 
valueB���*    *
dtype0*!
_output_shapes
:���
�
FC1/Variable/Adam_1
VariableV2"/device:GPU:0*!
_output_shapes
:���*
shared_name *
_class
loc:@FC1/Variable*
	container *
shape:���*
dtype0
�
FC1/Variable/Adam_1/AssignAssignFC1/Variable/Adam_1%FC1/Variable/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*
_class
loc:@FC1/Variable*
validate_shape(*!
_output_shapes
:���
�
FC1/Variable/Adam_1/readIdentityFC1/Variable/Adam_1"/device:GPU:0*
T0*
_class
loc:@FC1/Variable*!
_output_shapes
:���
�
%FC1/Variable_1/Adam/Initializer/zerosConst"/device:GPU:0*!
_class
loc:@FC1/Variable_1*
valueB�*    *
dtype0*
_output_shapes	
:�
�
FC1/Variable_1/Adam
VariableV2"/device:GPU:0*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *!
_class
loc:@FC1/Variable_1
�
FC1/Variable_1/Adam/AssignAssignFC1/Variable_1/Adam%FC1/Variable_1/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*!
_class
loc:@FC1/Variable_1*
validate_shape(*
_output_shapes	
:�
�
FC1/Variable_1/Adam/readIdentityFC1/Variable_1/Adam"/device:GPU:0*
T0*!
_class
loc:@FC1/Variable_1*
_output_shapes	
:�
�
'FC1/Variable_1/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes	
:�*!
_class
loc:@FC1/Variable_1*
valueB�*    *
dtype0
�
FC1/Variable_1/Adam_1
VariableV2"/device:GPU:0*!
_class
loc:@FC1/Variable_1*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
FC1/Variable_1/Adam_1/AssignAssignFC1/Variable_1/Adam_1'FC1/Variable_1/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*!
_class
loc:@FC1/Variable_1*
validate_shape(*
_output_shapes	
:�
�
FC1/Variable_1/Adam_1/readIdentityFC1/Variable_1/Adam_1"/device:GPU:0*
_output_shapes	
:�*
T0*!
_class
loc:@FC1/Variable_1
�
2batch_normalization_3/gamma/Adam/Initializer/zerosConst"/device:GPU:0*.
_class$
" loc:@batch_normalization_3/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
 batch_normalization_3/gamma/Adam
VariableV2"/device:GPU:0*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
'batch_normalization_3/gamma/Adam/AssignAssign batch_normalization_3/gamma/Adam2batch_normalization_3/gamma/Adam/Initializer/zeros"/device:GPU:0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
%batch_normalization_3/gamma/Adam/readIdentity batch_normalization_3/gamma/Adam"/device:GPU:0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:�*
T0
�
4batch_normalization_3/gamma/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes	
:�*.
_class$
" loc:@batch_normalization_3/gamma*
valueB�*    *
dtype0
�
"batch_normalization_3/gamma/Adam_1
VariableV2"/device:GPU:0*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
)batch_normalization_3/gamma/Adam_1/AssignAssign"batch_normalization_3/gamma/Adam_14batch_normalization_3/gamma/Adam_1/Initializer/zeros"/device:GPU:0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
'batch_normalization_3/gamma/Adam_1/readIdentity"batch_normalization_3/gamma/Adam_1"/device:GPU:0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:�*
T0
�
1batch_normalization_3/beta/Adam/Initializer/zerosConst"/device:GPU:0*-
_class#
!loc:@batch_normalization_3/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
batch_normalization_3/beta/Adam
VariableV2"/device:GPU:0*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
&batch_normalization_3/beta/Adam/AssignAssignbatch_normalization_3/beta/Adam1batch_normalization_3/beta/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:�
�
$batch_normalization_3/beta/Adam/readIdentitybatch_normalization_3/beta/Adam"/device:GPU:0*
_output_shapes	
:�*
T0*-
_class#
!loc:@batch_normalization_3/beta
�
3batch_normalization_3/beta/Adam_1/Initializer/zerosConst"/device:GPU:0*-
_class#
!loc:@batch_normalization_3/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
!batch_normalization_3/beta/Adam_1
VariableV2"/device:GPU:0*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
(batch_normalization_3/beta/Adam_1/AssignAssign!batch_normalization_3/beta/Adam_13batch_normalization_3/beta/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:�
�
&batch_normalization_3/beta/Adam_1/readIdentity!batch_normalization_3/beta/Adam_1"/device:GPU:0*
_output_shapes	
:�*
T0*-
_class#
!loc:@batch_normalization_3/beta
�
'Readout/Variable/Adam/Initializer/zerosConst"/device:GPU:0*#
_class
loc:@Readout/Variable*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
Readout/Variable/Adam
VariableV2"/device:GPU:0*
shared_name *#
_class
loc:@Readout/Variable*
	container *
shape:	�*
dtype0*
_output_shapes
:	�
�
Readout/Variable/Adam/AssignAssignReadout/Variable/Adam'Readout/Variable/Adam/Initializer/zeros"/device:GPU:0*#
_class
loc:@Readout/Variable*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0
�
Readout/Variable/Adam/readIdentityReadout/Variable/Adam"/device:GPU:0*
T0*#
_class
loc:@Readout/Variable*
_output_shapes
:	�
�
)Readout/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:	�*#
_class
loc:@Readout/Variable*
valueB	�*    *
dtype0
�
Readout/Variable/Adam_1
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:	�*
shared_name *#
_class
loc:@Readout/Variable*
	container *
shape:	�
�
Readout/Variable/Adam_1/AssignAssignReadout/Variable/Adam_1)Readout/Variable/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*#
_class
loc:@Readout/Variable*
validate_shape(*
_output_shapes
:	�
�
Readout/Variable/Adam_1/readIdentityReadout/Variable/Adam_1"/device:GPU:0*
T0*#
_class
loc:@Readout/Variable*
_output_shapes
:	�
�
)Readout/Variable_1/Adam/Initializer/zerosConst"/device:GPU:0*%
_class
loc:@Readout/Variable_1*
valueB*    *
dtype0*
_output_shapes
:
�
Readout/Variable_1/Adam
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:*
shared_name *%
_class
loc:@Readout/Variable_1*
	container *
shape:
�
Readout/Variable_1/Adam/AssignAssignReadout/Variable_1/Adam)Readout/Variable_1/Adam/Initializer/zeros"/device:GPU:0*%
_class
loc:@Readout/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
Readout/Variable_1/Adam/readIdentityReadout/Variable_1/Adam"/device:GPU:0*
_output_shapes
:*
T0*%
_class
loc:@Readout/Variable_1
�
+Readout/Variable_1/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:*%
_class
loc:@Readout/Variable_1*
valueB*    *
dtype0
�
Readout/Variable_1/Adam_1
VariableV2"/device:GPU:0*%
_class
loc:@Readout/Variable_1*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
 Readout/Variable_1/Adam_1/AssignAssignReadout/Variable_1/Adam_1+Readout/Variable_1/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*%
_class
loc:@Readout/Variable_1*
validate_shape(*
_output_shapes
:
�
Readout/Variable_1/Adam_1/readIdentityReadout/Variable_1/Adam_1"/device:GPU:0*
T0*%
_class
loc:@Readout/Variable_1*
_output_shapes
:
�
train/Adam/learning_rateConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *��8*
dtype0*
_output_shapes
: 
�
train/Adam/beta1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
train/Adam/beta2Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/Adam/epsilonConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
1train/Adam/update_CNN1/weights/Variable/ApplyAdam	ApplyAdamCNN1/weights/VariableCNN1/weights/Variable/AdamCNN1/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/CNN1/Conv2D_grad/tuple/control_dependency_1"/device:GPU:0*&
_output_shapes
:*
use_locking( *
T0*(
_class
loc:@CNN1/weights/Variable*
use_nesterov( 
�
0train/Adam/update_CNN1/biases/Variable/ApplyAdam	ApplyAdamCNN1/biases/VariableCNN1/biases/Variable/AdamCNN1/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon8train/gradients/CNN1/add_grad/tuple/control_dependency_1"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
5train/Adam/update_batch_normalization/gamma/ApplyAdam	ApplyAdambatch_normalization/gammabatch_normalization/gamma/Adam batch_normalization/gamma/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_9"/device:GPU:0*
use_locking( *
T0*,
_class"
 loc:@batch_normalization/gamma*
use_nesterov( *
_output_shapes
:@
�
4train/Adam/update_batch_normalization/beta/ApplyAdam	ApplyAdambatch_normalization/betabatch_normalization/beta/Adambatch_normalization/beta/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_10"/device:GPU:0*
_output_shapes
:@*
use_locking( *
T0*+
_class!
loc:@batch_normalization/beta*
use_nesterov( 
�
1train/Adam/update_CNN2/weights/Variable/ApplyAdam	ApplyAdamCNN2/weights/VariableCNN2/weights/Variable/AdamCNN2/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/CNN2/Conv2D_grad/tuple/control_dependency_1"/device:GPU:0*
use_locking( *
T0*(
_class
loc:@CNN2/weights/Variable*
use_nesterov( *&
_output_shapes
: 
�
0train/Adam/update_CNN2/biases/Variable/ApplyAdam	ApplyAdamCNN2/biases/VariableCNN2/biases/Variable/AdamCNN2/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon8train/gradients/CNN2/add_grad/tuple/control_dependency_1"/device:GPU:0*'
_class
loc:@CNN2/biases/Variable*
use_nesterov( *
_output_shapes
: *
use_locking( *
T0
�
7train/Adam/update_batch_normalization_1/gamma/ApplyAdam	ApplyAdambatch_normalization_1/gamma batch_normalization_1/gamma/Adam"batch_normalization_1/gamma/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_6"/device:GPU:0*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_1/gamma*
use_nesterov( *
_output_shapes
: 
�
6train/Adam/update_batch_normalization_1/beta/ApplyAdam	ApplyAdambatch_normalization_1/betabatch_normalization_1/beta/Adam!batch_normalization_1/beta/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_7"/device:GPU:0*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_1/beta*
use_nesterov( *
_output_shapes
: 
�
1train/Adam/update_CNN3/weights/Variable/ApplyAdam	ApplyAdamCNN3/weights/VariableCNN3/weights/Variable/AdamCNN3/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/CNN3/Conv2D_grad/tuple/control_dependency_1"/device:GPU:0*&
_output_shapes
: @*
use_locking( *
T0*(
_class
loc:@CNN3/weights/Variable*
use_nesterov( 
�
0train/Adam/update_CNN3/biases/Variable/ApplyAdam	ApplyAdamCNN3/biases/VariableCNN3/biases/Variable/AdamCNN3/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon8train/gradients/CNN3/add_grad/tuple/control_dependency_1"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0
�
7train/Adam/update_batch_normalization_2/gamma/ApplyAdam	ApplyAdambatch_normalization_2/gamma batch_normalization_2/gamma/Adam"batch_normalization_2/gamma/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_3"/device:GPU:0*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_2/gamma*
use_nesterov( *
_output_shapes
:
�
6train/Adam/update_batch_normalization_2/beta/ApplyAdam	ApplyAdambatch_normalization_2/betabatch_normalization_2/beta/Adam!batch_normalization_2/beta/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_4"/device:GPU:0*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_2/beta*
use_nesterov( *
_output_shapes
:
�
(train/Adam/update_FC1/Variable/ApplyAdam	ApplyAdamFC1/VariableFC1/Variable/AdamFC1/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/FC1/MatMul_grad/tuple/control_dependency_1"/device:GPU:0*!
_output_shapes
:���*
use_locking( *
T0*
_class
loc:@FC1/Variable*
use_nesterov( 
�
*train/Adam/update_FC1/Variable_1/ApplyAdam	ApplyAdamFC1/Variable_1FC1/Variable_1/AdamFC1/Variable_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon7train/gradients/FC1/add_grad/tuple/control_dependency_1"/device:GPU:0*!
_class
loc:@FC1/Variable_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
7train/Adam/update_batch_normalization_3/gamma/ApplyAdam	ApplyAdambatch_normalization_3/gamma batch_normalization_3/gamma/Adam"batch_normalization_3/gamma/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1"/device:GPU:0*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_3/gamma*
use_nesterov( *
_output_shapes	
:�
�
6train/Adam/update_batch_normalization_3/beta/ApplyAdam	ApplyAdambatch_normalization_3/betabatch_normalization_3/beta/Adam!batch_normalization_3/beta/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonStrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependency"/device:GPU:0*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_3/beta*
use_nesterov( *
_output_shapes	
:�
�
,train/Adam/update_Readout/Variable/ApplyAdam	ApplyAdamReadout/VariableReadout/Variable/AdamReadout/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/Readout/MatMul_grad/tuple/control_dependency_1"/device:GPU:0*
use_locking( *
T0*#
_class
loc:@Readout/Variable*
use_nesterov( *
_output_shapes
:	�
�
.train/Adam/update_Readout/Variable_1/ApplyAdam	ApplyAdamReadout/Variable_1Readout/Variable_1/AdamReadout/Variable_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/Readout/predicted_grad/tuple/control_dependency_1"/device:GPU:0*
_output_shapes
:*
use_locking( *
T0*%
_class
loc:@Readout/Variable_1*
use_nesterov( 
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta12^train/Adam/update_CNN1/weights/Variable/ApplyAdam1^train/Adam/update_CNN1/biases/Variable/ApplyAdam6^train/Adam/update_batch_normalization/gamma/ApplyAdam5^train/Adam/update_batch_normalization/beta/ApplyAdam2^train/Adam/update_CNN2/weights/Variable/ApplyAdam1^train/Adam/update_CNN2/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_1/gamma/ApplyAdam7^train/Adam/update_batch_normalization_1/beta/ApplyAdam2^train/Adam/update_CNN3/weights/Variable/ApplyAdam1^train/Adam/update_CNN3/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_2/gamma/ApplyAdam7^train/Adam/update_batch_normalization_2/beta/ApplyAdam)^train/Adam/update_FC1/Variable/ApplyAdam+^train/Adam/update_FC1/Variable_1/ApplyAdam8^train/Adam/update_batch_normalization_3/gamma/ApplyAdam7^train/Adam/update_batch_normalization_3/beta/ApplyAdam-^train/Adam/update_Readout/Variable/ApplyAdam/^train/Adam/update_Readout/Variable_1/ApplyAdam"/device:GPU:0*
_output_shapes
: *
T0*'
_class
loc:@CNN1/biases/Variable
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul"/device:GPU:0*
use_locking( *
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta22^train/Adam/update_CNN1/weights/Variable/ApplyAdam1^train/Adam/update_CNN1/biases/Variable/ApplyAdam6^train/Adam/update_batch_normalization/gamma/ApplyAdam5^train/Adam/update_batch_normalization/beta/ApplyAdam2^train/Adam/update_CNN2/weights/Variable/ApplyAdam1^train/Adam/update_CNN2/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_1/gamma/ApplyAdam7^train/Adam/update_batch_normalization_1/beta/ApplyAdam2^train/Adam/update_CNN3/weights/Variable/ApplyAdam1^train/Adam/update_CNN3/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_2/gamma/ApplyAdam7^train/Adam/update_batch_normalization_2/beta/ApplyAdam)^train/Adam/update_FC1/Variable/ApplyAdam+^train/Adam/update_FC1/Variable_1/ApplyAdam8^train/Adam/update_batch_normalization_3/gamma/ApplyAdam7^train/Adam/update_batch_normalization_3/beta/ApplyAdam-^train/Adam/update_Readout/Variable/ApplyAdam/^train/Adam/update_Readout/Variable_1/ApplyAdam"/device:GPU:0*
_output_shapes
: *
T0*'
_class
loc:@CNN1/biases/Variable
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1"/device:GPU:0*
use_locking( *
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
: 
�


train/AdamNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_12^train/Adam/update_CNN1/weights/Variable/ApplyAdam1^train/Adam/update_CNN1/biases/Variable/ApplyAdam6^train/Adam/update_batch_normalization/gamma/ApplyAdam5^train/Adam/update_batch_normalization/beta/ApplyAdam2^train/Adam/update_CNN2/weights/Variable/ApplyAdam1^train/Adam/update_CNN2/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_1/gamma/ApplyAdam7^train/Adam/update_batch_normalization_1/beta/ApplyAdam2^train/Adam/update_CNN3/weights/Variable/ApplyAdam1^train/Adam/update_CNN3/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_2/gamma/ApplyAdam7^train/Adam/update_batch_normalization_2/beta/ApplyAdam)^train/Adam/update_FC1/Variable/ApplyAdam+^train/Adam/update_FC1/Variable_1/ApplyAdam8^train/Adam/update_batch_normalization_3/gamma/ApplyAdam7^train/Adam/update_batch_normalization_3/beta/ApplyAdam-^train/Adam/update_Readout/Variable/ApplyAdam/^train/Adam/update_Readout/Variable_1/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1"/device:GPU:0
j
accuracy/ArgMax/dimensionConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
accuracy/ArgMaxArgMaxReadout/predictedaccuracy/ArgMax/dimension"/device:GPU:0*

Tidx0*
T0*
output_type0	*#
_output_shapes
:���������
l
accuracy/ArgMax_1/dimensionConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
accuracy/ArgMax_1ArgMaxinput/correct_labelsaccuracy/ArgMax_1/dimension"/device:GPU:0*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0

accuracy/correct_predEqualaccuracy/ArgMaxaccuracy/ArgMax_1"/device:GPU:0*
T0	*#
_output_shapes
:���������
x
accuracy/CastCastaccuracy/correct_pred"/device:GPU:0*

SrcT0
*#
_output_shapes
:���������*

DstT0
g
accuracy/ConstConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
accuracy/accuracyMeanaccuracy/Castaccuracy/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
4accuracy_streaming_mean/mean/total/Initializer/zerosConst*5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"accuracy_streaming_mean/mean/total
VariableV2"/device:GPU:0*
shared_name *5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
	container *
shape: *
dtype0*
_output_shapes
: 
�
)accuracy_streaming_mean/mean/total/AssignAssign"accuracy_streaming_mean/mean/total4accuracy_streaming_mean/mean/total/Initializer/zeros"/device:GPU:0*
_output_shapes
: *
use_locking(*
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
validate_shape(
�
'accuracy_streaming_mean/mean/total/readIdentity"accuracy_streaming_mean/mean/total"/device:GPU:0*
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
_output_shapes
: 
�
4accuracy_streaming_mean/mean/count/Initializer/zerosConst*5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"accuracy_streaming_mean/mean/count
VariableV2"/device:GPU:0*
shared_name *5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
	container *
shape: *
dtype0*
_output_shapes
: 
�
)accuracy_streaming_mean/mean/count/AssignAssign"accuracy_streaming_mean/mean/count4accuracy_streaming_mean/mean/count/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
validate_shape(*
_output_shapes
: 
�
'accuracy_streaming_mean/mean/count/readIdentity"accuracy_streaming_mean/mean/count"/device:GPU:0*
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
_output_shapes
: 
r
!accuracy_streaming_mean/mean/SizeConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
&accuracy_streaming_mean/mean/ToFloat_1Cast!accuracy_streaming_mean/mean/Size"/device:GPU:0*

SrcT0*
_output_shapes
: *

DstT0
t
"accuracy_streaming_mean/mean/ConstConst"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
 accuracy_streaming_mean/mean/SumSumaccuracy/accuracy"accuracy_streaming_mean/mean/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
&accuracy_streaming_mean/mean/AssignAdd	AssignAdd"accuracy_streaming_mean/mean/total accuracy_streaming_mean/mean/Sum"/device:GPU:0*5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
_output_shapes
: *
use_locking( *
T0
�
(accuracy_streaming_mean/mean/AssignAdd_1	AssignAdd"accuracy_streaming_mean/mean/count&accuracy_streaming_mean/mean/ToFloat_1^accuracy/accuracy"/device:GPU:0*
use_locking( *
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
_output_shapes
: 
z
&accuracy_streaming_mean/mean/Greater/yConst"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
$accuracy_streaming_mean/mean/GreaterGreater'accuracy_streaming_mean/mean/count/read&accuracy_streaming_mean/mean/Greater/y"/device:GPU:0*
T0*
_output_shapes
: 
�
$accuracy_streaming_mean/mean/truedivRealDiv'accuracy_streaming_mean/mean/total/read'accuracy_streaming_mean/mean/count/read"/device:GPU:0*
T0*
_output_shapes
: 
x
$accuracy_streaming_mean/mean/value/eConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"accuracy_streaming_mean/mean/valueSelect$accuracy_streaming_mean/mean/Greater$accuracy_streaming_mean/mean/truediv$accuracy_streaming_mean/mean/value/e"/device:GPU:0*
T0*
_output_shapes
: 
|
(accuracy_streaming_mean/mean/Greater_1/yConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
&accuracy_streaming_mean/mean/Greater_1Greater(accuracy_streaming_mean/mean/AssignAdd_1(accuracy_streaming_mean/mean/Greater_1/y"/device:GPU:0*
T0*
_output_shapes
: 
�
&accuracy_streaming_mean/mean/truediv_1RealDiv&accuracy_streaming_mean/mean/AssignAdd(accuracy_streaming_mean/mean/AssignAdd_1"/device:GPU:0*
_output_shapes
: *
T0
|
(accuracy_streaming_mean/mean/update_op/eConst"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
&accuracy_streaming_mean/mean/update_opSelect&accuracy_streaming_mean/mean/Greater_1&accuracy_streaming_mean/mean/truediv_1(accuracy_streaming_mean/mean/update_op/e"/device:GPU:0*
_output_shapes
: *
T0
�
accuracy_streaming_mean/initNoOp*^accuracy_streaming_mean/mean/total/Assign*^accuracy_streaming_mean/mean/count/Assign"/device:GPU:0
�
accuracy_streaming_mean_1/tagsConst"/device:GPU:0**
value!B Baccuracy_streaming_mean_1*
dtype0*
_output_shapes
: 
�
accuracy_streaming_mean_1ScalarSummaryaccuracy_streaming_mean_1/tags"accuracy_streaming_mean/mean/value"/device:GPU:0*
_output_shapes
: *
T0
�
Merge/MergeSummaryMergeSummaryCNN1/weights/summaries/meanCNN1/weights/summaries/stddev_1CNN1/weights/summaries/maxCNN1/weights/summaries/min CNN1/weights/summaries/histogramCNN1/biases/summaries/meanCNN1/biases/summaries/stddev_1CNN1/biases/summaries/maxCNN1/biases/summaries/minCNN1/biases/summaries/histogramCNN1/activationsCNN1/batch_normCNN2/weights/summaries/meanCNN2/weights/summaries/stddev_1CNN2/weights/summaries/maxCNN2/weights/summaries/min CNN2/weights/summaries/histogramCNN2/biases/summaries/meanCNN2/biases/summaries/stddev_1CNN2/biases/summaries/maxCNN2/biases/summaries/minCNN2/biases/summaries/histogramCNN2/activationsCNN2/batch_normCNN3/weights/summaries/meanCNN3/weights/summaries/stddev_1CNN3/weights/summaries/maxCNN3/weights/summaries/min CNN3/weights/summaries/histogramCNN3/biases/summaries/meanCNN3/biases/summaries/stddev_1CNN3/biases/summaries/maxCNN3/biases/summaries/minCNN3/biases/summaries/histogramCNN3/activationsCNN3/batch_normaccuracy_streaming_mean_1"/device:GPU:0*
N%*
_output_shapes
: "�3�![     �Á	1XOI���AJ��
�1�1
9
Add
x"T
y"T
z"T"
Ttype:
2	
T
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
�
ArgMax

input"T
	dimension"Tidx
output"output_type"
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
p
	AssignAdd
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
p
	AssignSub
ref"T�

value"T

output_ref"T�"
Ttype:
2	"
use_lockingbool( 
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
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
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
A
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
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
epsilonfloat%��8"
data_formatstringNHWC"
is_trainingbool(
�
FusedBatchNormGrad

y_backprop"T
x"T

scale"T
reserve_space_1"T
reserve_space_2"T

x_backprop"T
scale_backprop"T
offset_backprop"T
reserve_space_3"T
reserve_space_4"T"
Ttype:
2"
epsilonfloat%��8"
data_formatstringNHWC"
is_trainingbool(
:
Greater
x"T
y"T
z
"
Ttype:
2		
S
HistogramSummary
tag
values"T
summary"
Ttype0:
2		
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
�
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
-
Rsqrt
x"T
y"T"
Ttype:	
2
:
	RsqrtGrad
y"T
dy"T
z"T"
Ttype:	
2
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
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
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
i
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
,
Sqrt
x"T
y"T"
Ttype:	
2
0
Square
x"T
y"T"
Ttype:
	2	
F
SquaredDifference
x"T
y"T
z"T"
Ttype:
	2	�
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
9
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.4.12v1.4.0-19-ga52c8d9��
�
input/imagesPlaceholder"/device:GPU:0*
dtype0*/
_output_shapes
:���������@@*$
shape:���������@@
�
input/correct_labelsPlaceholder"/device:GPU:0*'
_output_shapes
:���������*
shape:���������*
dtype0
e
input/PlaceholderPlaceholder"/device:GPU:0*
_output_shapes
:*
shape:*
dtype0

�
#CNN1/weights/truncated_normal/shapeConst"/device:GPU:0*
_output_shapes
:*%
valueB"            *
dtype0
v
"CNN1/weights/truncated_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
x
$CNN1/weights/truncated_normal/stddevConst"/device:GPU:0*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
-CNN1/weights/truncated_normal/TruncatedNormalTruncatedNormal#CNN1/weights/truncated_normal/shape"/device:GPU:0*
T0*
dtype0*&
_output_shapes
:*
seed2 *

seed 
�
!CNN1/weights/truncated_normal/mulMul-CNN1/weights/truncated_normal/TruncatedNormal$CNN1/weights/truncated_normal/stddev"/device:GPU:0*
T0*&
_output_shapes
:
�
CNN1/weights/truncated_normalAdd!CNN1/weights/truncated_normal/mul"CNN1/weights/truncated_normal/mean"/device:GPU:0*&
_output_shapes
:*
T0
�
CNN1/weights/Variable
VariableV2"/device:GPU:0*
dtype0*&
_output_shapes
:*
	container *
shape:*
shared_name 
�
CNN1/weights/Variable/AssignAssignCNN1/weights/VariableCNN1/weights/truncated_normal"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN1/weights/Variable*
validate_shape(*&
_output_shapes
:
�
CNN1/weights/Variable/readIdentityCNN1/weights/Variable"/device:GPU:0*(
_class
loc:@CNN1/weights/Variable*&
_output_shapes
:*
T0
l
CNN1/weights/summaries/RankConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
s
"CNN1/weights/summaries/range/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
s
"CNN1/weights/summaries/range/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN1/weights/summaries/rangeRange"CNN1/weights/summaries/range/startCNN1/weights/summaries/Rank"CNN1/weights/summaries/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/weights/summaries/MeanMeanCNN1/weights/Variable/readCNN1/weights/summaries/range"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
 CNN1/weights/summaries/mean/tagsConst"/device:GPU:0*,
value#B! BCNN1/weights/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN1/weights/summaries/meanScalarSummary CNN1/weights/summaries/mean/tagsCNN1/weights/summaries/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
!CNN1/weights/summaries/stddev/subSubCNN1/weights/Variable/readCNN1/weights/summaries/Mean"/device:GPU:0*&
_output_shapes
:*
T0
�
$CNN1/weights/summaries/stddev/SquareSquare!CNN1/weights/summaries/stddev/sub"/device:GPU:0*
T0*&
_output_shapes
:
�
#CNN1/weights/summaries/stddev/ConstConst"/device:GPU:0*
_output_shapes
:*%
valueB"             *
dtype0
�
"CNN1/weights/summaries/stddev/MeanMean$CNN1/weights/summaries/stddev/Square#CNN1/weights/summaries/stddev/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
~
"CNN1/weights/summaries/stddev/SqrtSqrt"CNN1/weights/summaries/stddev/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
$CNN1/weights/summaries/stddev_1/tagsConst"/device:GPU:0*0
value'B% BCNN1/weights/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
CNN1/weights/summaries/stddev_1ScalarSummary$CNN1/weights/summaries/stddev_1/tags"CNN1/weights/summaries/stddev/Sqrt"/device:GPU:0*
_output_shapes
: *
T0
n
CNN1/weights/summaries/Rank_1Const"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
u
$CNN1/weights/summaries/range_1/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
u
$CNN1/weights/summaries/range_1/deltaConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
CNN1/weights/summaries/range_1Range$CNN1/weights/summaries/range_1/startCNN1/weights/summaries/Rank_1$CNN1/weights/summaries/range_1/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/weights/summaries/MaxMaxCNN1/weights/Variable/readCNN1/weights/summaries/range_1"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
CNN1/weights/summaries/max/tagsConst"/device:GPU:0*
_output_shapes
: *+
value"B  BCNN1/weights/summaries/max*
dtype0
�
CNN1/weights/summaries/maxScalarSummaryCNN1/weights/summaries/max/tagsCNN1/weights/summaries/Max"/device:GPU:0*
_output_shapes
: *
T0
n
CNN1/weights/summaries/Rank_2Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
u
$CNN1/weights/summaries/range_2/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
u
$CNN1/weights/summaries/range_2/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN1/weights/summaries/range_2Range$CNN1/weights/summaries/range_2/startCNN1/weights/summaries/Rank_2$CNN1/weights/summaries/range_2/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/weights/summaries/MinMinCNN1/weights/Variable/readCNN1/weights/summaries/range_2"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN1/weights/summaries/min/tagsConst"/device:GPU:0*
_output_shapes
: *+
value"B  BCNN1/weights/summaries/min*
dtype0
�
CNN1/weights/summaries/minScalarSummaryCNN1/weights/summaries/min/tagsCNN1/weights/summaries/Min"/device:GPU:0*
_output_shapes
: *
T0
�
$CNN1/weights/summaries/histogram/tagConst"/device:GPU:0*1
value(B& B CNN1/weights/summaries/histogram*
dtype0*
_output_shapes
: 
�
 CNN1/weights/summaries/histogramHistogramSummary$CNN1/weights/summaries/histogram/tagCNN1/weights/Variable/read"/device:GPU:0*
T0*
_output_shapes
: 
m
CNN1/biases/ConstConst"/device:GPU:0*
valueB*���=*
dtype0*
_output_shapes
:
�
CNN1/biases/Variable
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
CNN1/biases/Variable/AssignAssignCNN1/biases/VariableCNN1/biases/Const"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
:
�
CNN1/biases/Variable/readIdentityCNN1/biases/Variable"/device:GPU:0*
_output_shapes
:*
T0*'
_class
loc:@CNN1/biases/Variable
k
CNN1/biases/summaries/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
r
!CNN1/biases/summaries/range/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
r
!CNN1/biases/summaries/range/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/rangeRange!CNN1/biases/summaries/range/startCNN1/biases/summaries/Rank!CNN1/biases/summaries/range/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
�
CNN1/biases/summaries/MeanMeanCNN1/biases/Variable/readCNN1/biases/summaries/range"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
CNN1/biases/summaries/mean/tagsConst"/device:GPU:0*+
value"B  BCNN1/biases/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/meanScalarSummaryCNN1/biases/summaries/mean/tagsCNN1/biases/summaries/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
 CNN1/biases/summaries/stddev/subSubCNN1/biases/Variable/readCNN1/biases/summaries/Mean"/device:GPU:0*
T0*
_output_shapes
:
�
#CNN1/biases/summaries/stddev/SquareSquare CNN1/biases/summaries/stddev/sub"/device:GPU:0*
_output_shapes
:*
T0
{
"CNN1/biases/summaries/stddev/ConstConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
!CNN1/biases/summaries/stddev/MeanMean#CNN1/biases/summaries/stddev/Square"CNN1/biases/summaries/stddev/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
|
!CNN1/biases/summaries/stddev/SqrtSqrt!CNN1/biases/summaries/stddev/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
#CNN1/biases/summaries/stddev_1/tagsConst"/device:GPU:0*/
value&B$ BCNN1/biases/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/stddev_1ScalarSummary#CNN1/biases/summaries/stddev_1/tags!CNN1/biases/summaries/stddev/Sqrt"/device:GPU:0*
_output_shapes
: *
T0
m
CNN1/biases/summaries/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
t
#CNN1/biases/summaries/range_1/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
t
#CNN1/biases/summaries/range_1/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/range_1Range#CNN1/biases/summaries/range_1/startCNN1/biases/summaries/Rank_1#CNN1/biases/summaries/range_1/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/biases/summaries/MaxMaxCNN1/biases/Variable/readCNN1/biases/summaries/range_1"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN1/biases/summaries/max/tagsConst"/device:GPU:0**
value!B BCNN1/biases/summaries/max*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/maxScalarSummaryCNN1/biases/summaries/max/tagsCNN1/biases/summaries/Max"/device:GPU:0*
T0*
_output_shapes
: 
m
CNN1/biases/summaries/Rank_2Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
t
#CNN1/biases/summaries/range_2/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
t
#CNN1/biases/summaries/range_2/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/range_2Range#CNN1/biases/summaries/range_2/startCNN1/biases/summaries/Rank_2#CNN1/biases/summaries/range_2/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/biases/summaries/MinMinCNN1/biases/Variable/readCNN1/biases/summaries/range_2"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN1/biases/summaries/min/tagsConst"/device:GPU:0**
value!B BCNN1/biases/summaries/min*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/minScalarSummaryCNN1/biases/summaries/min/tagsCNN1/biases/summaries/Min"/device:GPU:0*
_output_shapes
: *
T0
�
#CNN1/biases/summaries/histogram/tagConst"/device:GPU:0*0
value'B% BCNN1/biases/summaries/histogram*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/histogramHistogramSummary#CNN1/biases/summaries/histogram/tagCNN1/biases/Variable/read"/device:GPU:0*
_output_shapes
: *
T0
�
CNN1/Conv2DConv2Dinput/imagesCNN1/weights/Variable/read"/device:GPU:0*/
_output_shapes
:���������@@*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
CNN1/addAddCNN1/Conv2DCNN1/biases/Variable/read"/device:GPU:0*/
_output_shapes
:���������@@*
T0
d
	CNN1/ReluReluCNN1/add"/device:GPU:0*
T0*/
_output_shapes
:���������@@
t
CNN1/activations/tagConst"/device:GPU:0*!
valueB BCNN1/activations*
dtype0*
_output_shapes
: 
u
CNN1/activationsHistogramSummaryCNN1/activations/tag	CNN1/Relu"/device:GPU:0*
_output_shapes
: *
T0
�
*batch_normalization/gamma/Initializer/onesConst*,
_class"
 loc:@batch_normalization/gamma*
valueB@*  �?*
dtype0*
_output_shapes
:@
�
batch_normalization/gamma
VariableV2"/device:GPU:0*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *,
_class"
 loc:@batch_normalization/gamma
�
 batch_normalization/gamma/AssignAssignbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones"/device:GPU:0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
batch_normalization/gamma/readIdentitybatch_normalization/gamma"/device:GPU:0*
_output_shapes
:@*
T0*,
_class"
 loc:@batch_normalization/gamma
�
*batch_normalization/beta/Initializer/zerosConst*+
_class!
loc:@batch_normalization/beta*
valueB@*    *
dtype0*
_output_shapes
:@
�
batch_normalization/beta
VariableV2"/device:GPU:0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container *
shape:@*
dtype0
�
batch_normalization/beta/AssignAssignbatch_normalization/beta*batch_normalization/beta/Initializer/zeros"/device:GPU:0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
batch_normalization/beta/readIdentitybatch_normalization/beta"/device:GPU:0*
_output_shapes
:@*
T0*+
_class!
loc:@batch_normalization/beta
�
1batch_normalization/moving_mean/Initializer/zerosConst*
_output_shapes
:@*2
_class(
&$loc:@batch_normalization/moving_mean*
valueB@*    *
dtype0
�
batch_normalization/moving_mean
VariableV2"/device:GPU:0*
_output_shapes
:@*
shared_name *2
_class(
&$loc:@batch_normalization/moving_mean*
	container *
shape:@*
dtype0
�
&batch_normalization/moving_mean/AssignAssignbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros"/device:GPU:0*2
_class(
&$loc:@batch_normalization/moving_mean*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
$batch_normalization/moving_mean/readIdentitybatch_normalization/moving_mean"/device:GPU:0*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
�
4batch_normalization/moving_variance/Initializer/onesConst*6
_class,
*(loc:@batch_normalization/moving_variance*
valueB@*  �?*
dtype0*
_output_shapes
:@
�
#batch_normalization/moving_variance
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:@*
shared_name *6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape:@
�
*batch_normalization/moving_variance/AssignAssign#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones"/device:GPU:0*6
_class,
*(loc:@batch_normalization/moving_variance*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
(batch_normalization/moving_variance/readIdentity#batch_normalization/moving_variance"/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
$CNN1/batch_normalization/cond/SwitchSwitchinput/Placeholderinput/Placeholder"/device:GPU:0*
T0
*
_output_shapes

::
�
&CNN1/batch_normalization/cond/switch_tIdentity&CNN1/batch_normalization/cond/Switch:1"/device:GPU:0*
_output_shapes
:*
T0

�
&CNN1/batch_normalization/cond/switch_fIdentity$CNN1/batch_normalization/cond/Switch"/device:GPU:0*
T0
*
_output_shapes
:
v
%CNN1/batch_normalization/cond/pred_idIdentityinput/Placeholder"/device:GPU:0*
_output_shapes
:*
T0

�
#CNN1/batch_normalization/cond/ConstConst'^CNN1/batch_normalization/cond/switch_t"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
%CNN1/batch_normalization/cond/Const_1Const'^CNN1/batch_normalization/cond/switch_t"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
3CNN1/batch_normalization/cond/FusedBatchNorm/SwitchSwitch	CNN1/Relu%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*J
_output_shapes8
6:���������@@:���������@@*
T0*
_class
loc:@CNN1/Relu
�
5CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1Switchbatch_normalization/gamma/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*,
_class"
 loc:@batch_normalization/gamma* 
_output_shapes
:@:@*
T0
�
5CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization/beta/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*
T0*+
_class!
loc:@batch_normalization/beta* 
_output_shapes
:@:@
�
,CNN1/batch_normalization/cond/FusedBatchNormFusedBatchNorm5CNN1/batch_normalization/cond/FusedBatchNorm/Switch:17CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1:17CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2:1#CNN1/batch_normalization/cond/Const%CNN1/batch_normalization/cond/Const_1"/device:GPU:0*
data_formatNCHW*G
_output_shapes5
3:���������@@:@:@:@:@*
is_training(*
epsilon%o�:*
T0
�
5CNN1/batch_normalization/cond/FusedBatchNorm_1/SwitchSwitch	CNN1/Relu%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*
T0*
_class
loc:@CNN1/Relu*J
_output_shapes8
6:���������@@:���������@@
�
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1Switchbatch_normalization/gamma/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*
T0*,
_class"
 loc:@batch_normalization/gamma* 
_output_shapes
:@:@
�
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization/beta/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*
T0*+
_class!
loc:@batch_normalization/beta* 
_output_shapes
:@:@
�
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch$batch_normalization/moving_mean/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*2
_class(
&$loc:@batch_normalization/moving_mean* 
_output_shapes
:@:@*
T0
�
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch(batch_normalization/moving_variance/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
:@:@*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
.CNN1/batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm5CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_27CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
data_formatNCHW*G
_output_shapes5
3:���������@@:@:@:@:@*
is_training( *
epsilon%o�:*
T0
�
#CNN1/batch_normalization/cond/MergeMerge.CNN1/batch_normalization/cond/FusedBatchNorm_1,CNN1/batch_normalization/cond/FusedBatchNorm"/device:GPU:0*
T0*
N*1
_output_shapes
:���������@@: 
�
%CNN1/batch_normalization/cond/Merge_1Merge0CNN1/batch_normalization/cond/FusedBatchNorm_1:1.CNN1/batch_normalization/cond/FusedBatchNorm:1"/device:GPU:0*
_output_shapes

:@: *
T0*
N
�
%CNN1/batch_normalization/cond/Merge_2Merge0CNN1/batch_normalization/cond/FusedBatchNorm_1:2.CNN1/batch_normalization/cond/FusedBatchNorm:2"/device:GPU:0*
N*
_output_shapes

:@: *
T0
}
)CNN1/batch_normalization/ExpandDims/inputConst"/device:GPU:0*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
x
'CNN1/batch_normalization/ExpandDims/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
#CNN1/batch_normalization/ExpandDims
ExpandDims)CNN1/batch_normalization/ExpandDims/input'CNN1/batch_normalization/ExpandDims/dim"/device:GPU:0*

Tdim0*
T0*
_output_shapes
:

+CNN1/batch_normalization/ExpandDims_1/inputConst"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
z
)CNN1/batch_normalization/ExpandDims_1/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
%CNN1/batch_normalization/ExpandDims_1
ExpandDims+CNN1/batch_normalization/ExpandDims_1/input)CNN1/batch_normalization/ExpandDims_1/dim"/device:GPU:0*
T0*
_output_shapes
:*

Tdim0

&CNN1/batch_normalization/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
 CNN1/batch_normalization/ReshapeReshapeinput/Placeholder&CNN1/batch_normalization/Reshape/shape"/device:GPU:0*
T0
*
Tshape0*
_output_shapes
:
�
CNN1/batch_normalization/SelectSelect CNN1/batch_normalization/Reshape#CNN1/batch_normalization/ExpandDims%CNN1/batch_normalization/ExpandDims_1"/device:GPU:0*
_output_shapes
:*
T0
�
 CNN1/batch_normalization/SqueezeSqueezeCNN1/batch_normalization/Select"/device:GPU:0*
_output_shapes
: *
squeeze_dims
 *
T0
�
-CNN1/batch_normalization/AssignMovingAvg/readIdentitybatch_normalization/moving_mean"/device:GPU:0*
_output_shapes
:@*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
,CNN1/batch_normalization/AssignMovingAvg/SubSub-CNN1/batch_normalization/AssignMovingAvg/read%CNN1/batch_normalization/cond/Merge_1"/device:GPU:0*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
�
,CNN1/batch_normalization/AssignMovingAvg/MulMul,CNN1/batch_normalization/AssignMovingAvg/Sub CNN1/batch_normalization/Squeeze"/device:GPU:0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@*
T0
�
(CNN1/batch_normalization/AssignMovingAvg	AssignSubbatch_normalization/moving_mean,CNN1/batch_normalization/AssignMovingAvg/Mul"/device:GPU:0*
use_locking( *
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
�
/CNN1/batch_normalization/AssignMovingAvg_1/readIdentity#batch_normalization/moving_variance"/device:GPU:0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@*
T0
�
.CNN1/batch_normalization/AssignMovingAvg_1/SubSub/CNN1/batch_normalization/AssignMovingAvg_1/read%CNN1/batch_normalization/cond/Merge_2"/device:GPU:0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@*
T0
�
.CNN1/batch_normalization/AssignMovingAvg_1/MulMul.CNN1/batch_normalization/AssignMovingAvg_1/Sub CNN1/batch_normalization/Squeeze"/device:GPU:0*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
�
*CNN1/batch_normalization/AssignMovingAvg_1	AssignSub#batch_normalization/moving_variance.CNN1/batch_normalization/AssignMovingAvg_1/Mul"/device:GPU:0*
use_locking( *
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
r
CNN1/batch_norm/tagConst"/device:GPU:0* 
valueB BCNN1/batch_norm*
dtype0*
_output_shapes
: 
�
CNN1/batch_normHistogramSummaryCNN1/batch_norm/tag#CNN1/batch_normalization/cond/Merge"/device:GPU:0*
T0*
_output_shapes
: 
�
#CNN2/weights/truncated_normal/shapeConst"/device:GPU:0*
_output_shapes
:*%
valueB"             *
dtype0
v
"CNN2/weights/truncated_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
x
$CNN2/weights/truncated_normal/stddevConst"/device:GPU:0*
_output_shapes
: *
valueB
 *���=*
dtype0
�
-CNN2/weights/truncated_normal/TruncatedNormalTruncatedNormal#CNN2/weights/truncated_normal/shape"/device:GPU:0*&
_output_shapes
: *
seed2 *

seed *
T0*
dtype0
�
!CNN2/weights/truncated_normal/mulMul-CNN2/weights/truncated_normal/TruncatedNormal$CNN2/weights/truncated_normal/stddev"/device:GPU:0*&
_output_shapes
: *
T0
�
CNN2/weights/truncated_normalAdd!CNN2/weights/truncated_normal/mul"CNN2/weights/truncated_normal/mean"/device:GPU:0*&
_output_shapes
: *
T0
�
CNN2/weights/Variable
VariableV2"/device:GPU:0*
dtype0*&
_output_shapes
: *
	container *
shape: *
shared_name 
�
CNN2/weights/Variable/AssignAssignCNN2/weights/VariableCNN2/weights/truncated_normal"/device:GPU:0*&
_output_shapes
: *
use_locking(*
T0*(
_class
loc:@CNN2/weights/Variable*
validate_shape(
�
CNN2/weights/Variable/readIdentityCNN2/weights/Variable"/device:GPU:0*
T0*(
_class
loc:@CNN2/weights/Variable*&
_output_shapes
: 
l
CNN2/weights/summaries/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
s
"CNN2/weights/summaries/range/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
s
"CNN2/weights/summaries/range/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/rangeRange"CNN2/weights/summaries/range/startCNN2/weights/summaries/Rank"CNN2/weights/summaries/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN2/weights/summaries/MeanMeanCNN2/weights/Variable/readCNN2/weights/summaries/range"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
 CNN2/weights/summaries/mean/tagsConst"/device:GPU:0*,
value#B! BCNN2/weights/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/meanScalarSummary CNN2/weights/summaries/mean/tagsCNN2/weights/summaries/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
!CNN2/weights/summaries/stddev/subSubCNN2/weights/Variable/readCNN2/weights/summaries/Mean"/device:GPU:0*
T0*&
_output_shapes
: 
�
$CNN2/weights/summaries/stddev/SquareSquare!CNN2/weights/summaries/stddev/sub"/device:GPU:0*
T0*&
_output_shapes
: 
�
#CNN2/weights/summaries/stddev/ConstConst"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
"CNN2/weights/summaries/stddev/MeanMean$CNN2/weights/summaries/stddev/Square#CNN2/weights/summaries/stddev/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
~
"CNN2/weights/summaries/stddev/SqrtSqrt"CNN2/weights/summaries/stddev/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
$CNN2/weights/summaries/stddev_1/tagsConst"/device:GPU:0*0
value'B% BCNN2/weights/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/stddev_1ScalarSummary$CNN2/weights/summaries/stddev_1/tags"CNN2/weights/summaries/stddev/Sqrt"/device:GPU:0*
T0*
_output_shapes
: 
n
CNN2/weights/summaries/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
u
$CNN2/weights/summaries/range_1/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
u
$CNN2/weights/summaries/range_1/deltaConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
CNN2/weights/summaries/range_1Range$CNN2/weights/summaries/range_1/startCNN2/weights/summaries/Rank_1$CNN2/weights/summaries/range_1/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN2/weights/summaries/MaxMaxCNN2/weights/Variable/readCNN2/weights/summaries/range_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN2/weights/summaries/max/tagsConst"/device:GPU:0*+
value"B  BCNN2/weights/summaries/max*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/maxScalarSummaryCNN2/weights/summaries/max/tagsCNN2/weights/summaries/Max"/device:GPU:0*
T0*
_output_shapes
: 
n
CNN2/weights/summaries/Rank_2Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
u
$CNN2/weights/summaries/range_2/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
u
$CNN2/weights/summaries/range_2/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/range_2Range$CNN2/weights/summaries/range_2/startCNN2/weights/summaries/Rank_2$CNN2/weights/summaries/range_2/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
�
CNN2/weights/summaries/MinMinCNN2/weights/Variable/readCNN2/weights/summaries/range_2"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN2/weights/summaries/min/tagsConst"/device:GPU:0*
_output_shapes
: *+
value"B  BCNN2/weights/summaries/min*
dtype0
�
CNN2/weights/summaries/minScalarSummaryCNN2/weights/summaries/min/tagsCNN2/weights/summaries/Min"/device:GPU:0*
T0*
_output_shapes
: 
�
$CNN2/weights/summaries/histogram/tagConst"/device:GPU:0*
_output_shapes
: *1
value(B& B CNN2/weights/summaries/histogram*
dtype0
�
 CNN2/weights/summaries/histogramHistogramSummary$CNN2/weights/summaries/histogram/tagCNN2/weights/Variable/read"/device:GPU:0*
T0*
_output_shapes
: 
m
CNN2/biases/ConstConst"/device:GPU:0*
valueB *���=*
dtype0*
_output_shapes
: 
�
CNN2/biases/Variable
VariableV2"/device:GPU:0*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
CNN2/biases/Variable/AssignAssignCNN2/biases/VariableCNN2/biases/Const"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN2/biases/Variable*
validate_shape(*
_output_shapes
: 
�
CNN2/biases/Variable/readIdentityCNN2/biases/Variable"/device:GPU:0*'
_class
loc:@CNN2/biases/Variable*
_output_shapes
: *
T0
k
CNN2/biases/summaries/RankConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
r
!CNN2/biases/summaries/range/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
r
!CNN2/biases/summaries/range/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/rangeRange!CNN2/biases/summaries/range/startCNN2/biases/summaries/Rank!CNN2/biases/summaries/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN2/biases/summaries/MeanMeanCNN2/biases/Variable/readCNN2/biases/summaries/range"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
CNN2/biases/summaries/mean/tagsConst"/device:GPU:0*+
value"B  BCNN2/biases/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/meanScalarSummaryCNN2/biases/summaries/mean/tagsCNN2/biases/summaries/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
 CNN2/biases/summaries/stddev/subSubCNN2/biases/Variable/readCNN2/biases/summaries/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
#CNN2/biases/summaries/stddev/SquareSquare CNN2/biases/summaries/stddev/sub"/device:GPU:0*
T0*
_output_shapes
: 
{
"CNN2/biases/summaries/stddev/ConstConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
!CNN2/biases/summaries/stddev/MeanMean#CNN2/biases/summaries/stddev/Square"CNN2/biases/summaries/stddev/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
|
!CNN2/biases/summaries/stddev/SqrtSqrt!CNN2/biases/summaries/stddev/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
#CNN2/biases/summaries/stddev_1/tagsConst"/device:GPU:0*/
value&B$ BCNN2/biases/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/stddev_1ScalarSummary#CNN2/biases/summaries/stddev_1/tags!CNN2/biases/summaries/stddev/Sqrt"/device:GPU:0*
_output_shapes
: *
T0
m
CNN2/biases/summaries/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
t
#CNN2/biases/summaries/range_1/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
t
#CNN2/biases/summaries/range_1/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/range_1Range#CNN2/biases/summaries/range_1/startCNN2/biases/summaries/Rank_1#CNN2/biases/summaries/range_1/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN2/biases/summaries/MaxMaxCNN2/biases/Variable/readCNN2/biases/summaries/range_1"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN2/biases/summaries/max/tagsConst"/device:GPU:0**
value!B BCNN2/biases/summaries/max*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/maxScalarSummaryCNN2/biases/summaries/max/tagsCNN2/biases/summaries/Max"/device:GPU:0*
_output_shapes
: *
T0
m
CNN2/biases/summaries/Rank_2Const"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
t
#CNN2/biases/summaries/range_2/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
t
#CNN2/biases/summaries/range_2/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/range_2Range#CNN2/biases/summaries/range_2/startCNN2/biases/summaries/Rank_2#CNN2/biases/summaries/range_2/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
�
CNN2/biases/summaries/MinMinCNN2/biases/Variable/readCNN2/biases/summaries/range_2"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN2/biases/summaries/min/tagsConst"/device:GPU:0*
_output_shapes
: **
value!B BCNN2/biases/summaries/min*
dtype0
�
CNN2/biases/summaries/minScalarSummaryCNN2/biases/summaries/min/tagsCNN2/biases/summaries/Min"/device:GPU:0*
_output_shapes
: *
T0
�
#CNN2/biases/summaries/histogram/tagConst"/device:GPU:0*0
value'B% BCNN2/biases/summaries/histogram*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/histogramHistogramSummary#CNN2/biases/summaries/histogram/tagCNN2/biases/Variable/read"/device:GPU:0*
T0*
_output_shapes
: 
�
CNN2/Conv2DConv2D#CNN1/batch_normalization/cond/MergeCNN2/weights/Variable/read"/device:GPU:0*/
_output_shapes
:���������   *
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
CNN2/addAddCNN2/Conv2DCNN2/biases/Variable/read"/device:GPU:0*/
_output_shapes
:���������   *
T0
d
	CNN2/ReluReluCNN2/add"/device:GPU:0*
T0*/
_output_shapes
:���������   
t
CNN2/activations/tagConst"/device:GPU:0*!
valueB BCNN2/activations*
dtype0*
_output_shapes
: 
u
CNN2/activationsHistogramSummaryCNN2/activations/tag	CNN2/Relu"/device:GPU:0*
_output_shapes
: *
T0
�
,batch_normalization_1/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_1/gamma*
valueB *  �?*
dtype0*
_output_shapes
: 
�
batch_normalization_1/gamma
VariableV2"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones"/device:GPU:0*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes
: 
�
 batch_normalization_1/gamma/readIdentitybatch_normalization_1/gamma"/device:GPU:0*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_1/gamma
�
,batch_normalization_1/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_1/beta*
valueB *    *
dtype0*
_output_shapes
: 
�
batch_normalization_1/beta
VariableV2"/device:GPU:0*
_output_shapes
: *
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape: *
dtype0
�
!batch_normalization_1/beta/AssignAssignbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes
: 
�
batch_normalization_1/beta/readIdentitybatch_normalization_1/beta"/device:GPU:0*
_output_shapes
: *
T0*-
_class#
!loc:@batch_normalization_1/beta
�
3batch_normalization_1/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
valueB *    *
dtype0*
_output_shapes
: 
�
!batch_normalization_1/moving_mean
VariableV2"/device:GPU:0*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@batch_normalization_1/moving_mean
�
(batch_normalization_1/moving_mean/AssignAssign!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes
: 
�
&batch_normalization_1/moving_mean/readIdentity!batch_normalization_1/moving_mean"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
�
6batch_normalization_1/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB *  �?*
dtype0*
_output_shapes
: 
�
%batch_normalization_1/moving_variance
VariableV2"/device:GPU:0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones"/device:GPU:0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
*batch_normalization_1/moving_variance/readIdentity%batch_normalization_1/moving_variance"/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
$CNN2/batch_normalization/cond/SwitchSwitchinput/Placeholderinput/Placeholder"/device:GPU:0*
T0
*
_output_shapes

::
�
&CNN2/batch_normalization/cond/switch_tIdentity&CNN2/batch_normalization/cond/Switch:1"/device:GPU:0*
_output_shapes
:*
T0

�
&CNN2/batch_normalization/cond/switch_fIdentity$CNN2/batch_normalization/cond/Switch"/device:GPU:0*
T0
*
_output_shapes
:
v
%CNN2/batch_normalization/cond/pred_idIdentityinput/Placeholder"/device:GPU:0*
_output_shapes
:*
T0

�
#CNN2/batch_normalization/cond/ConstConst'^CNN2/batch_normalization/cond/switch_t"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
%CNN2/batch_normalization/cond/Const_1Const'^CNN2/batch_normalization/cond/switch_t"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
3CNN2/batch_normalization/cond/FusedBatchNorm/SwitchSwitch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*
_class
loc:@CNN2/Relu*J
_output_shapes8
6:���������   :���������   
�
5CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1Switch batch_normalization_1/gamma/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
: : *
T0
�
5CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
: : *
T0*-
_class#
!loc:@batch_normalization_1/beta
�
,CNN2/batch_normalization/cond/FusedBatchNormFusedBatchNorm5CNN2/batch_normalization/cond/FusedBatchNorm/Switch:17CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1:17CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2:1#CNN2/batch_normalization/cond/Const%CNN2/batch_normalization/cond/Const_1"/device:GPU:0*
data_formatNCHW*G
_output_shapes5
3:���������   : : : : *
is_training(*
epsilon%o�:*
T0
�
5CNN2/batch_normalization/cond/FusedBatchNorm_1/SwitchSwitch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
_class
loc:@CNN2/Relu*J
_output_shapes8
6:���������   :���������   *
T0
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_1/gamma/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
: : 
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
: : 
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_1/moving_mean/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean* 
_output_shapes
: : 
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_1/moving_variance/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*8
_class.
,*loc:@batch_normalization_1/moving_variance* 
_output_shapes
: : *
T0
�
.CNN2/batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm5CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_27CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*G
_output_shapes5
3:���������   : : : : *
is_training( *
epsilon%o�:*
T0*
data_formatNCHW
�
#CNN2/batch_normalization/cond/MergeMerge.CNN2/batch_normalization/cond/FusedBatchNorm_1,CNN2/batch_normalization/cond/FusedBatchNorm"/device:GPU:0*
N*1
_output_shapes
:���������   : *
T0
�
%CNN2/batch_normalization/cond/Merge_1Merge0CNN2/batch_normalization/cond/FusedBatchNorm_1:1.CNN2/batch_normalization/cond/FusedBatchNorm:1"/device:GPU:0*
N*
_output_shapes

: : *
T0
�
%CNN2/batch_normalization/cond/Merge_2Merge0CNN2/batch_normalization/cond/FusedBatchNorm_1:2.CNN2/batch_normalization/cond/FusedBatchNorm:2"/device:GPU:0*
N*
_output_shapes

: : *
T0
}
)CNN2/batch_normalization/ExpandDims/inputConst"/device:GPU:0*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
x
'CNN2/batch_normalization/ExpandDims/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
#CNN2/batch_normalization/ExpandDims
ExpandDims)CNN2/batch_normalization/ExpandDims/input'CNN2/batch_normalization/ExpandDims/dim"/device:GPU:0*
T0*
_output_shapes
:*

Tdim0

+CNN2/batch_normalization/ExpandDims_1/inputConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
z
)CNN2/batch_normalization/ExpandDims_1/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
%CNN2/batch_normalization/ExpandDims_1
ExpandDims+CNN2/batch_normalization/ExpandDims_1/input)CNN2/batch_normalization/ExpandDims_1/dim"/device:GPU:0*
T0*
_output_shapes
:*

Tdim0

&CNN2/batch_normalization/Reshape/shapeConst"/device:GPU:0*
_output_shapes
:*
valueB:*
dtype0
�
 CNN2/batch_normalization/ReshapeReshapeinput/Placeholder&CNN2/batch_normalization/Reshape/shape"/device:GPU:0*
_output_shapes
:*
T0
*
Tshape0
�
CNN2/batch_normalization/SelectSelect CNN2/batch_normalization/Reshape#CNN2/batch_normalization/ExpandDims%CNN2/batch_normalization/ExpandDims_1"/device:GPU:0*
_output_shapes
:*
T0
�
 CNN2/batch_normalization/SqueezeSqueezeCNN2/batch_normalization/Select"/device:GPU:0*
_output_shapes
: *
squeeze_dims
 *
T0
�
-CNN2/batch_normalization/AssignMovingAvg/readIdentity!batch_normalization_1/moving_mean"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
�
,CNN2/batch_normalization/AssignMovingAvg/SubSub-CNN2/batch_normalization/AssignMovingAvg/read%CNN2/batch_normalization/cond/Merge_1"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
�
,CNN2/batch_normalization/AssignMovingAvg/MulMul,CNN2/batch_normalization/AssignMovingAvg/Sub CNN2/batch_normalization/Squeeze"/device:GPU:0*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
(CNN2/batch_normalization/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean,CNN2/batch_normalization/AssignMovingAvg/Mul"/device:GPU:0*
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
�
/CNN2/batch_normalization/AssignMovingAvg_1/readIdentity%batch_normalization_1/moving_variance"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
�
.CNN2/batch_normalization/AssignMovingAvg_1/SubSub/CNN2/batch_normalization/AssignMovingAvg_1/read%CNN2/batch_normalization/cond/Merge_2"/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
.CNN2/batch_normalization/AssignMovingAvg_1/MulMul.CNN2/batch_normalization/AssignMovingAvg_1/Sub CNN2/batch_normalization/Squeeze"/device:GPU:0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: *
T0
�
*CNN2/batch_normalization/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance.CNN2/batch_normalization/AssignMovingAvg_1/Mul"/device:GPU:0*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
r
CNN2/batch_norm/tagConst"/device:GPU:0* 
valueB BCNN2/batch_norm*
dtype0*
_output_shapes
: 
�
CNN2/batch_normHistogramSummaryCNN2/batch_norm/tag#CNN2/batch_normalization/cond/Merge"/device:GPU:0*
T0*
_output_shapes
: 
�
#CNN3/weights/truncated_normal/shapeConst"/device:GPU:0*%
valueB"          @   *
dtype0*
_output_shapes
:
v
"CNN3/weights/truncated_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
x
$CNN3/weights/truncated_normal/stddevConst"/device:GPU:0*
_output_shapes
: *
valueB
 *���=*
dtype0
�
-CNN3/weights/truncated_normal/TruncatedNormalTruncatedNormal#CNN3/weights/truncated_normal/shape"/device:GPU:0*
dtype0*&
_output_shapes
: @*
seed2 *

seed *
T0
�
!CNN3/weights/truncated_normal/mulMul-CNN3/weights/truncated_normal/TruncatedNormal$CNN3/weights/truncated_normal/stddev"/device:GPU:0*
T0*&
_output_shapes
: @
�
CNN3/weights/truncated_normalAdd!CNN3/weights/truncated_normal/mul"CNN3/weights/truncated_normal/mean"/device:GPU:0*&
_output_shapes
: @*
T0
�
CNN3/weights/Variable
VariableV2"/device:GPU:0*
dtype0*&
_output_shapes
: @*
	container *
shape: @*
shared_name 
�
CNN3/weights/Variable/AssignAssignCNN3/weights/VariableCNN3/weights/truncated_normal"/device:GPU:0*(
_class
loc:@CNN3/weights/Variable*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0
�
CNN3/weights/Variable/readIdentityCNN3/weights/Variable"/device:GPU:0*(
_class
loc:@CNN3/weights/Variable*&
_output_shapes
: @*
T0
l
CNN3/weights/summaries/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
s
"CNN3/weights/summaries/range/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
s
"CNN3/weights/summaries/range/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/rangeRange"CNN3/weights/summaries/range/startCNN3/weights/summaries/Rank"CNN3/weights/summaries/range/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
�
CNN3/weights/summaries/MeanMeanCNN3/weights/Variable/readCNN3/weights/summaries/range"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
 CNN3/weights/summaries/mean/tagsConst"/device:GPU:0*,
value#B! BCNN3/weights/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/meanScalarSummary CNN3/weights/summaries/mean/tagsCNN3/weights/summaries/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
!CNN3/weights/summaries/stddev/subSubCNN3/weights/Variable/readCNN3/weights/summaries/Mean"/device:GPU:0*&
_output_shapes
: @*
T0
�
$CNN3/weights/summaries/stddev/SquareSquare!CNN3/weights/summaries/stddev/sub"/device:GPU:0*
T0*&
_output_shapes
: @
�
#CNN3/weights/summaries/stddev/ConstConst"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
"CNN3/weights/summaries/stddev/MeanMean$CNN3/weights/summaries/stddev/Square#CNN3/weights/summaries/stddev/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
~
"CNN3/weights/summaries/stddev/SqrtSqrt"CNN3/weights/summaries/stddev/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
$CNN3/weights/summaries/stddev_1/tagsConst"/device:GPU:0*0
value'B% BCNN3/weights/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/stddev_1ScalarSummary$CNN3/weights/summaries/stddev_1/tags"CNN3/weights/summaries/stddev/Sqrt"/device:GPU:0*
_output_shapes
: *
T0
n
CNN3/weights/summaries/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
u
$CNN3/weights/summaries/range_1/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
u
$CNN3/weights/summaries/range_1/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/range_1Range$CNN3/weights/summaries/range_1/startCNN3/weights/summaries/Rank_1$CNN3/weights/summaries/range_1/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN3/weights/summaries/MaxMaxCNN3/weights/Variable/readCNN3/weights/summaries/range_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN3/weights/summaries/max/tagsConst"/device:GPU:0*+
value"B  BCNN3/weights/summaries/max*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/maxScalarSummaryCNN3/weights/summaries/max/tagsCNN3/weights/summaries/Max"/device:GPU:0*
T0*
_output_shapes
: 
n
CNN3/weights/summaries/Rank_2Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
u
$CNN3/weights/summaries/range_2/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
u
$CNN3/weights/summaries/range_2/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/range_2Range$CNN3/weights/summaries/range_2/startCNN3/weights/summaries/Rank_2$CNN3/weights/summaries/range_2/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN3/weights/summaries/MinMinCNN3/weights/Variable/readCNN3/weights/summaries/range_2"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN3/weights/summaries/min/tagsConst"/device:GPU:0*+
value"B  BCNN3/weights/summaries/min*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/minScalarSummaryCNN3/weights/summaries/min/tagsCNN3/weights/summaries/Min"/device:GPU:0*
T0*
_output_shapes
: 
�
$CNN3/weights/summaries/histogram/tagConst"/device:GPU:0*1
value(B& B CNN3/weights/summaries/histogram*
dtype0*
_output_shapes
: 
�
 CNN3/weights/summaries/histogramHistogramSummary$CNN3/weights/summaries/histogram/tagCNN3/weights/Variable/read"/device:GPU:0*
_output_shapes
: *
T0
m
CNN3/biases/ConstConst"/device:GPU:0*
_output_shapes
:@*
valueB@*���=*
dtype0
�
CNN3/biases/Variable
VariableV2"/device:GPU:0*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
�
CNN3/biases/Variable/AssignAssignCNN3/biases/VariableCNN3/biases/Const"/device:GPU:0*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@CNN3/biases/Variable*
validate_shape(
�
CNN3/biases/Variable/readIdentityCNN3/biases/Variable"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
_output_shapes
:@*
T0
k
CNN3/biases/summaries/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
r
!CNN3/biases/summaries/range/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
r
!CNN3/biases/summaries/range/deltaConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
CNN3/biases/summaries/rangeRange!CNN3/biases/summaries/range/startCNN3/biases/summaries/Rank!CNN3/biases/summaries/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN3/biases/summaries/MeanMeanCNN3/biases/Variable/readCNN3/biases/summaries/range"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
CNN3/biases/summaries/mean/tagsConst"/device:GPU:0*+
value"B  BCNN3/biases/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/meanScalarSummaryCNN3/biases/summaries/mean/tagsCNN3/biases/summaries/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
 CNN3/biases/summaries/stddev/subSubCNN3/biases/Variable/readCNN3/biases/summaries/Mean"/device:GPU:0*
T0*
_output_shapes
:@
�
#CNN3/biases/summaries/stddev/SquareSquare CNN3/biases/summaries/stddev/sub"/device:GPU:0*
_output_shapes
:@*
T0
{
"CNN3/biases/summaries/stddev/ConstConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
!CNN3/biases/summaries/stddev/MeanMean#CNN3/biases/summaries/stddev/Square"CNN3/biases/summaries/stddev/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
|
!CNN3/biases/summaries/stddev/SqrtSqrt!CNN3/biases/summaries/stddev/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
#CNN3/biases/summaries/stddev_1/tagsConst"/device:GPU:0*/
value&B$ BCNN3/biases/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/stddev_1ScalarSummary#CNN3/biases/summaries/stddev_1/tags!CNN3/biases/summaries/stddev/Sqrt"/device:GPU:0*
T0*
_output_shapes
: 
m
CNN3/biases/summaries/Rank_1Const"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
t
#CNN3/biases/summaries/range_1/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
t
#CNN3/biases/summaries/range_1/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/range_1Range#CNN3/biases/summaries/range_1/startCNN3/biases/summaries/Rank_1#CNN3/biases/summaries/range_1/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
�
CNN3/biases/summaries/MaxMaxCNN3/biases/Variable/readCNN3/biases/summaries/range_1"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN3/biases/summaries/max/tagsConst"/device:GPU:0**
value!B BCNN3/biases/summaries/max*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/maxScalarSummaryCNN3/biases/summaries/max/tagsCNN3/biases/summaries/Max"/device:GPU:0*
_output_shapes
: *
T0
m
CNN3/biases/summaries/Rank_2Const"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
t
#CNN3/biases/summaries/range_2/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
t
#CNN3/biases/summaries/range_2/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/range_2Range#CNN3/biases/summaries/range_2/startCNN3/biases/summaries/Rank_2#CNN3/biases/summaries/range_2/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN3/biases/summaries/MinMinCNN3/biases/Variable/readCNN3/biases/summaries/range_2"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN3/biases/summaries/min/tagsConst"/device:GPU:0**
value!B BCNN3/biases/summaries/min*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/minScalarSummaryCNN3/biases/summaries/min/tagsCNN3/biases/summaries/Min"/device:GPU:0*
T0*
_output_shapes
: 
�
#CNN3/biases/summaries/histogram/tagConst"/device:GPU:0*0
value'B% BCNN3/biases/summaries/histogram*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/histogramHistogramSummary#CNN3/biases/summaries/histogram/tagCNN3/biases/Variable/read"/device:GPU:0*
T0*
_output_shapes
: 
�
CNN3/Conv2DConv2D#CNN2/batch_normalization/cond/MergeCNN3/weights/Variable/read"/device:GPU:0*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������@
�
CNN3/addAddCNN3/Conv2DCNN3/biases/Variable/read"/device:GPU:0*/
_output_shapes
:���������@*
T0
d
	CNN3/ReluReluCNN3/add"/device:GPU:0*/
_output_shapes
:���������@*
T0
t
CNN3/activations/tagConst"/device:GPU:0*!
valueB BCNN3/activations*
dtype0*
_output_shapes
: 
u
CNN3/activationsHistogramSummaryCNN3/activations/tag	CNN3/Relu"/device:GPU:0*
_output_shapes
: *
T0
�
,batch_normalization_2/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*  �?*
dtype0*
_output_shapes
:
�
batch_normalization_2/gamma
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:
�
"batch_normalization_2/gamma/AssignAssignbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(
�
 batch_normalization_2/gamma/readIdentitybatch_normalization_2/gamma"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
�
,batch_normalization_2/beta/Initializer/zerosConst*
_output_shapes
:*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
dtype0
�
batch_normalization_2/beta
VariableV2"/device:GPU:0*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_2/beta
�
!batch_normalization_2/beta/AssignAssignbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
:
�
batch_normalization_2/beta/readIdentitybatch_normalization_2/beta"/device:GPU:0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:*
T0
�
3batch_normalization_2/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_2/moving_mean*
valueB*    *
dtype0*
_output_shapes
:
�
!batch_normalization_2/moving_mean
VariableV2"/device:GPU:0*
shared_name *4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container *
shape:*
dtype0*
_output_shapes
:
�
(batch_normalization_2/moving_mean/AssignAssign!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros"/device:GPU:0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean"/device:GPU:0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:*
T0
�
6batch_normalization_2/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_2/moving_variance*
valueB*  �?*
dtype0*
_output_shapes
:
�
%batch_normalization_2/moving_variance
VariableV2"/device:GPU:0*
shared_name *8
_class.
,*loc:@batch_normalization_2/moving_variance*
	container *
shape:*
dtype0*
_output_shapes
:
�
,batch_normalization_2/moving_variance/AssignAssign%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
validate_shape(
�
*batch_normalization_2/moving_variance/readIdentity%batch_normalization_2/moving_variance"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
$CNN3/batch_normalization/cond/SwitchSwitchinput/Placeholderinput/Placeholder"/device:GPU:0*
_output_shapes

::*
T0

�
&CNN3/batch_normalization/cond/switch_tIdentity&CNN3/batch_normalization/cond/Switch:1"/device:GPU:0*
_output_shapes
:*
T0

�
&CNN3/batch_normalization/cond/switch_fIdentity$CNN3/batch_normalization/cond/Switch"/device:GPU:0*
_output_shapes
:*
T0

v
%CNN3/batch_normalization/cond/pred_idIdentityinput/Placeholder"/device:GPU:0*
_output_shapes
:*
T0

�
#CNN3/batch_normalization/cond/ConstConst'^CNN3/batch_normalization/cond/switch_t"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
%CNN3/batch_normalization/cond/Const_1Const'^CNN3/batch_normalization/cond/switch_t"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
3CNN3/batch_normalization/cond/FusedBatchNorm/SwitchSwitch	CNN3/Relu%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*
_class
loc:@CNN3/Relu*J
_output_shapes8
6:���������@:���������@
�
5CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_2/gamma* 
_output_shapes
::
�
5CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_2/beta
�
,CNN3/batch_normalization/cond/FusedBatchNormFusedBatchNorm5CNN3/batch_normalization/cond/FusedBatchNorm/Switch:17CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1:17CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2:1#CNN3/batch_normalization/cond/Const%CNN3/batch_normalization/cond/Const_1"/device:GPU:0*G
_output_shapes5
3:���������@::::*
is_training(*
epsilon%o�:*
T0*
data_formatNCHW
�
5CNN3/batch_normalization/cond/FusedBatchNorm_1/SwitchSwitch	CNN3/Relu%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*
_class
loc:@CNN3/Relu*J
_output_shapes8
6:���������@:���������@
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*.
_class$
" loc:@batch_normalization_2/gamma* 
_output_shapes
::*
T0
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
::*
T0
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_2/moving_mean/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*4
_class*
(&loc:@batch_normalization_2/moving_mean* 
_output_shapes
::*
T0
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_2/moving_variance/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance* 
_output_shapes
::
�
.CNN3/batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm5CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_27CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
T0*
data_formatNCHW*G
_output_shapes5
3:���������@::::*
is_training( *
epsilon%o�:
�
#CNN3/batch_normalization/cond/MergeMerge.CNN3/batch_normalization/cond/FusedBatchNorm_1,CNN3/batch_normalization/cond/FusedBatchNorm"/device:GPU:0*
T0*
N*1
_output_shapes
:���������@: 
�
%CNN3/batch_normalization/cond/Merge_1Merge0CNN3/batch_normalization/cond/FusedBatchNorm_1:1.CNN3/batch_normalization/cond/FusedBatchNorm:1"/device:GPU:0*
T0*
N*
_output_shapes

:: 
�
%CNN3/batch_normalization/cond/Merge_2Merge0CNN3/batch_normalization/cond/FusedBatchNorm_1:2.CNN3/batch_normalization/cond/FusedBatchNorm:2"/device:GPU:0*
T0*
N*
_output_shapes

:: 
}
)CNN3/batch_normalization/ExpandDims/inputConst"/device:GPU:0*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
x
'CNN3/batch_normalization/ExpandDims/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
#CNN3/batch_normalization/ExpandDims
ExpandDims)CNN3/batch_normalization/ExpandDims/input'CNN3/batch_normalization/ExpandDims/dim"/device:GPU:0*

Tdim0*
T0*
_output_shapes
:

+CNN3/batch_normalization/ExpandDims_1/inputConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
z
)CNN3/batch_normalization/ExpandDims_1/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
%CNN3/batch_normalization/ExpandDims_1
ExpandDims+CNN3/batch_normalization/ExpandDims_1/input)CNN3/batch_normalization/ExpandDims_1/dim"/device:GPU:0*

Tdim0*
T0*
_output_shapes
:

&CNN3/batch_normalization/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
 CNN3/batch_normalization/ReshapeReshapeinput/Placeholder&CNN3/batch_normalization/Reshape/shape"/device:GPU:0*
_output_shapes
:*
T0
*
Tshape0
�
CNN3/batch_normalization/SelectSelect CNN3/batch_normalization/Reshape#CNN3/batch_normalization/ExpandDims%CNN3/batch_normalization/ExpandDims_1"/device:GPU:0*
T0*
_output_shapes
:
�
 CNN3/batch_normalization/SqueezeSqueezeCNN3/batch_normalization/Select"/device:GPU:0*
T0*
_output_shapes
: *
squeeze_dims
 
�
-CNN3/batch_normalization/AssignMovingAvg/readIdentity!batch_normalization_2/moving_mean"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
,CNN3/batch_normalization/AssignMovingAvg/SubSub-CNN3/batch_normalization/AssignMovingAvg/read%CNN3/batch_normalization/cond/Merge_1"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
,CNN3/batch_normalization/AssignMovingAvg/MulMul,CNN3/batch_normalization/AssignMovingAvg/Sub CNN3/batch_normalization/Squeeze"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:
�
(CNN3/batch_normalization/AssignMovingAvg	AssignSub!batch_normalization_2/moving_mean,CNN3/batch_normalization/AssignMovingAvg/Mul"/device:GPU:0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:*
use_locking( *
T0
�
/CNN3/batch_normalization/AssignMovingAvg_1/readIdentity%batch_normalization_2/moving_variance"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
.CNN3/batch_normalization/AssignMovingAvg_1/SubSub/CNN3/batch_normalization/AssignMovingAvg_1/read%CNN3/batch_normalization/cond/Merge_2"/device:GPU:0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
T0
�
.CNN3/batch_normalization/AssignMovingAvg_1/MulMul.CNN3/batch_normalization/AssignMovingAvg_1/Sub CNN3/batch_normalization/Squeeze"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
*CNN3/batch_normalization/AssignMovingAvg_1	AssignSub%batch_normalization_2/moving_variance.CNN3/batch_normalization/AssignMovingAvg_1/Mul"/device:GPU:0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
use_locking( *
T0
r
CNN3/batch_norm/tagConst"/device:GPU:0* 
valueB BCNN3/batch_norm*
dtype0*
_output_shapes
: 
�
CNN3/batch_normHistogramSummaryCNN3/batch_norm/tag#CNN3/batch_normalization/cond/Merge"/device:GPU:0*
T0*
_output_shapes
: 
z
FC1/truncated_normal/shapeConst"/device:GPU:0*
_output_shapes
:*
valueB" @     *
dtype0
m
FC1/truncated_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
o
FC1/truncated_normal/stddevConst"/device:GPU:0*
_output_shapes
: *
valueB
 *���=*
dtype0
�
$FC1/truncated_normal/TruncatedNormalTruncatedNormalFC1/truncated_normal/shape"/device:GPU:0*
dtype0*!
_output_shapes
:���*
seed2 *

seed *
T0
�
FC1/truncated_normal/mulMul$FC1/truncated_normal/TruncatedNormalFC1/truncated_normal/stddev"/device:GPU:0*
T0*!
_output_shapes
:���
�
FC1/truncated_normalAddFC1/truncated_normal/mulFC1/truncated_normal/mean"/device:GPU:0*!
_output_shapes
:���*
T0
�
FC1/Variable
VariableV2"/device:GPU:0*
shape:���*
shared_name *
dtype0*!
_output_shapes
:���*
	container 
�
FC1/Variable/AssignAssignFC1/VariableFC1/truncated_normal"/device:GPU:0*
use_locking(*
T0*
_class
loc:@FC1/Variable*
validate_shape(*!
_output_shapes
:���
�
FC1/Variable/readIdentityFC1/Variable"/device:GPU:0*
T0*
_class
loc:@FC1/Variable*!
_output_shapes
:���
g
	FC1/ConstConst"/device:GPU:0*
valueB�*���=*
dtype0*
_output_shapes	
:�
�
FC1/Variable_1
VariableV2"/device:GPU:0*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
�
FC1/Variable_1/AssignAssignFC1/Variable_1	FC1/Const"/device:GPU:0*!
_class
loc:@FC1/Variable_1*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
FC1/Variable_1/readIdentityFC1/Variable_1"/device:GPU:0*
T0*!
_class
loc:@FC1/Variable_1*
_output_shapes	
:�
q
FC1/Reshape/shapeConst"/device:GPU:0*
valueB"���� @  *
dtype0*
_output_shapes
:
�
FC1/ReshapeReshape#CNN3/batch_normalization/cond/MergeFC1/Reshape/shape"/device:GPU:0*
T0*
Tshape0*)
_output_shapes
:�����������
�

FC1/MatMulMatMulFC1/ReshapeFC1/Variable/read"/device:GPU:0*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
q
FC1/addAdd
FC1/MatMulFC1/Variable_1/read"/device:GPU:0*(
_output_shapes
:����������*
T0
[
FC1/ReluReluFC1/add"/device:GPU:0*(
_output_shapes
:����������*
T0
�
,batch_normalization_3/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_3/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
batch_normalization_3/gamma
VariableV2"/device:GPU:0*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma
�
"batch_normalization_3/gamma/AssignAssignbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones"/device:GPU:0*
_output_shapes	
:�*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(
�
 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma"/device:GPU:0*
_output_shapes	
:�*
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
,batch_normalization_3/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_3/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
batch_normalization_3/beta
VariableV2"/device:GPU:0*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
!batch_normalization_3/beta/AssignAssignbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:�
�
batch_normalization_3/beta/readIdentitybatch_normalization_3/beta"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:�
�
3batch_normalization_3/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
!batch_normalization_3/moving_mean
VariableV2"/device:GPU:0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
validate_shape(*
_output_shapes	
:�
�
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean"/device:GPU:0*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
6batch_normalization_3/moving_variance/Initializer/onesConst*
_output_shapes	
:�*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB�*  �?*
dtype0
�
%batch_normalization_3/moving_variance
VariableV2"/device:GPU:0*
shared_name *8
_class.
,*loc:@batch_normalization_3/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones"/device:GPU:0*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
validate_shape(*
_output_shapes	
:�
�
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance"/device:GPU:0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:�*
T0
�
6FC1/batch_normalization/moments/mean/reduction_indicesConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
$FC1/batch_normalization/moments/meanMeanFC1/Relu6FC1/batch_normalization/moments/mean/reduction_indices"/device:GPU:0*

Tidx0*
	keep_dims(*
T0*
_output_shapes
:	�
�
,FC1/batch_normalization/moments/StopGradientStopGradient$FC1/batch_normalization/moments/mean"/device:GPU:0*
T0*
_output_shapes
:	�
�
1FC1/batch_normalization/moments/SquaredDifferenceSquaredDifferenceFC1/Relu,FC1/batch_normalization/moments/StopGradient"/device:GPU:0*(
_output_shapes
:����������*
T0
�
:FC1/batch_normalization/moments/variance/reduction_indicesConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
(FC1/batch_normalization/moments/varianceMean1FC1/batch_normalization/moments/SquaredDifference:FC1/batch_normalization/moments/variance/reduction_indices"/device:GPU:0*
_output_shapes
:	�*

Tidx0*
	keep_dims(*
T0
�
'FC1/batch_normalization/moments/SqueezeSqueeze$FC1/batch_normalization/moments/mean"/device:GPU:0*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
)FC1/batch_normalization/moments/Squeeze_1Squeeze(FC1/batch_normalization/moments/variance"/device:GPU:0*
_output_shapes	
:�*
squeeze_dims
 *
T0
w
&FC1/batch_normalization/ExpandDims/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
"FC1/batch_normalization/ExpandDims
ExpandDims'FC1/batch_normalization/moments/Squeeze&FC1/batch_normalization/ExpandDims/dim"/device:GPU:0*
_output_shapes
:	�*

Tdim0*
T0
y
(FC1/batch_normalization/ExpandDims_1/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
$FC1/batch_normalization/ExpandDims_1
ExpandDims&batch_normalization_3/moving_mean/read(FC1/batch_normalization/ExpandDims_1/dim"/device:GPU:0*
_output_shapes
:	�*

Tdim0*
T0
~
%FC1/batch_normalization/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
FC1/batch_normalization/ReshapeReshapeinput/Placeholder%FC1/batch_normalization/Reshape/shape"/device:GPU:0*
_output_shapes
:*
T0
*
Tshape0
�
FC1/batch_normalization/SelectSelectFC1/batch_normalization/Reshape"FC1/batch_normalization/ExpandDims$FC1/batch_normalization/ExpandDims_1"/device:GPU:0*
T0*
_output_shapes
:	�
�
FC1/batch_normalization/SqueezeSqueezeFC1/batch_normalization/Select"/device:GPU:0*
squeeze_dims
 *
T0*
_output_shapes	
:�
y
(FC1/batch_normalization/ExpandDims_2/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
$FC1/batch_normalization/ExpandDims_2
ExpandDims)FC1/batch_normalization/moments/Squeeze_1(FC1/batch_normalization/ExpandDims_2/dim"/device:GPU:0*

Tdim0*
T0*
_output_shapes
:	�
y
(FC1/batch_normalization/ExpandDims_3/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
$FC1/batch_normalization/ExpandDims_3
ExpandDims*batch_normalization_3/moving_variance/read(FC1/batch_normalization/ExpandDims_3/dim"/device:GPU:0*
_output_shapes
:	�*

Tdim0*
T0
�
'FC1/batch_normalization/Reshape_1/shapeConst"/device:GPU:0*
_output_shapes
:*
valueB:*
dtype0
�
!FC1/batch_normalization/Reshape_1Reshapeinput/Placeholder'FC1/batch_normalization/Reshape_1/shape"/device:GPU:0*
T0
*
Tshape0*
_output_shapes
:
�
 FC1/batch_normalization/Select_1Select!FC1/batch_normalization/Reshape_1$FC1/batch_normalization/ExpandDims_2$FC1/batch_normalization/ExpandDims_3"/device:GPU:0*
T0*
_output_shapes
:	�
�
!FC1/batch_normalization/Squeeze_1Squeeze FC1/batch_normalization/Select_1"/device:GPU:0*
squeeze_dims
 *
T0*
_output_shapes	
:�
~
*FC1/batch_normalization/ExpandDims_4/inputConst"/device:GPU:0*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
y
(FC1/batch_normalization/ExpandDims_4/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
$FC1/batch_normalization/ExpandDims_4
ExpandDims*FC1/batch_normalization/ExpandDims_4/input(FC1/batch_normalization/ExpandDims_4/dim"/device:GPU:0*
T0*
_output_shapes
:*

Tdim0
~
*FC1/batch_normalization/ExpandDims_5/inputConst"/device:GPU:0*
_output_shapes
: *
valueB
 *  �?*
dtype0
y
(FC1/batch_normalization/ExpandDims_5/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
$FC1/batch_normalization/ExpandDims_5
ExpandDims*FC1/batch_normalization/ExpandDims_5/input(FC1/batch_normalization/ExpandDims_5/dim"/device:GPU:0*
_output_shapes
:*

Tdim0*
T0
�
'FC1/batch_normalization/Reshape_2/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
!FC1/batch_normalization/Reshape_2Reshapeinput/Placeholder'FC1/batch_normalization/Reshape_2/shape"/device:GPU:0*
_output_shapes
:*
T0
*
Tshape0
�
 FC1/batch_normalization/Select_2Select!FC1/batch_normalization/Reshape_2$FC1/batch_normalization/ExpandDims_4$FC1/batch_normalization/ExpandDims_5"/device:GPU:0*
_output_shapes
:*
T0
�
!FC1/batch_normalization/Squeeze_2Squeeze FC1/batch_normalization/Select_2"/device:GPU:0*
_output_shapes
: *
squeeze_dims
 *
T0
�
-FC1/batch_normalization/AssignMovingAvg/sub/xConst"/device:GPU:0*
valueB
 *  �?*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
: 
�
+FC1/batch_normalization/AssignMovingAvg/subSub-FC1/batch_normalization/AssignMovingAvg/sub/x!FC1/batch_normalization/Squeeze_2"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: 
�
-FC1/batch_normalization/AssignMovingAvg/sub_1Sub&batch_normalization_3/moving_mean/readFC1/batch_normalization/Squeeze"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:�
�
+FC1/batch_normalization/AssignMovingAvg/mulMul-FC1/batch_normalization/AssignMovingAvg/sub_1+FC1/batch_normalization/AssignMovingAvg/sub"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:�
�
'FC1/batch_normalization/AssignMovingAvg	AssignSub!batch_normalization_3/moving_mean+FC1/batch_normalization/AssignMovingAvg/mul"/device:GPU:0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
/FC1/batch_normalization/AssignMovingAvg_1/sub/xConst"/device:GPU:0*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
: 
�
-FC1/batch_normalization/AssignMovingAvg_1/subSub/FC1/batch_normalization/AssignMovingAvg_1/sub/x!FC1/batch_normalization/Squeeze_2"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
: 
�
/FC1/batch_normalization/AssignMovingAvg_1/sub_1Sub*batch_normalization_3/moving_variance/read!FC1/batch_normalization/Squeeze_1"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:�
�
-FC1/batch_normalization/AssignMovingAvg_1/mulMul/FC1/batch_normalization/AssignMovingAvg_1/sub_1-FC1/batch_normalization/AssignMovingAvg_1/sub"/device:GPU:0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:�*
T0
�
)FC1/batch_normalization/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance-FC1/batch_normalization/AssignMovingAvg_1/mul"/device:GPU:0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
{
'FC1/batch_normalization/batchnorm/add/yConst"/device:GPU:0*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
%FC1/batch_normalization/batchnorm/addAdd!FC1/batch_normalization/Squeeze_1'FC1/batch_normalization/batchnorm/add/y"/device:GPU:0*
T0*
_output_shapes	
:�
�
'FC1/batch_normalization/batchnorm/RsqrtRsqrt%FC1/batch_normalization/batchnorm/add"/device:GPU:0*
T0*
_output_shapes	
:�
�
%FC1/batch_normalization/batchnorm/mulMul'FC1/batch_normalization/batchnorm/Rsqrt batch_normalization_3/gamma/read"/device:GPU:0*
T0*
_output_shapes	
:�
�
'FC1/batch_normalization/batchnorm/mul_1MulFC1/Relu%FC1/batch_normalization/batchnorm/mul"/device:GPU:0*(
_output_shapes
:����������*
T0
�
'FC1/batch_normalization/batchnorm/mul_2MulFC1/batch_normalization/Squeeze%FC1/batch_normalization/batchnorm/mul"/device:GPU:0*
_output_shapes	
:�*
T0
�
%FC1/batch_normalization/batchnorm/subSubbatch_normalization_3/beta/read'FC1/batch_normalization/batchnorm/mul_2"/device:GPU:0*
_output_shapes	
:�*
T0
�
'FC1/batch_normalization/batchnorm/add_1Add'FC1/batch_normalization/batchnorm/mul_1%FC1/batch_normalization/batchnorm/sub"/device:GPU:0*
T0*(
_output_shapes
:����������
h
dropout/dropout_probPlaceholder"/device:GPU:0*
shape:*
dtype0*
_output_shapes
:
~
Readout/truncated_normal/shapeConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
q
Readout/truncated_normal/meanConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
s
Readout/truncated_normal/stddevConst"/device:GPU:0*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
(Readout/truncated_normal/TruncatedNormalTruncatedNormalReadout/truncated_normal/shape"/device:GPU:0*
dtype0*
_output_shapes
:	�*
seed2 *

seed *
T0
�
Readout/truncated_normal/mulMul(Readout/truncated_normal/TruncatedNormalReadout/truncated_normal/stddev"/device:GPU:0*
T0*
_output_shapes
:	�
�
Readout/truncated_normalAddReadout/truncated_normal/mulReadout/truncated_normal/mean"/device:GPU:0*
T0*
_output_shapes
:	�
�
Readout/Variable
VariableV2"/device:GPU:0*
_output_shapes
:	�*
	container *
shape:	�*
shared_name *
dtype0
�
Readout/Variable/AssignAssignReadout/VariableReadout/truncated_normal"/device:GPU:0*#
_class
loc:@Readout/Variable*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0
�
Readout/Variable/readIdentityReadout/Variable"/device:GPU:0*
T0*#
_class
loc:@Readout/Variable*
_output_shapes
:	�
i
Readout/ConstConst"/device:GPU:0*
valueB*���=*
dtype0*
_output_shapes
:
�
Readout/Variable_1
VariableV2"/device:GPU:0*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
Readout/Variable_1/AssignAssignReadout/Variable_1Readout/Const"/device:GPU:0*
use_locking(*
T0*%
_class
loc:@Readout/Variable_1*
validate_shape(*
_output_shapes
:
�
Readout/Variable_1/readIdentityReadout/Variable_1"/device:GPU:0*
T0*%
_class
loc:@Readout/Variable_1*
_output_shapes
:
�
Readout/MatMulMatMul'FC1/batch_normalization/batchnorm/add_1Readout/Variable/read"/device:GPU:0*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
Readout/predictedAddReadout/MatMulReadout/Variable_1/read"/device:GPU:0*'
_output_shapes
:���������*
T0
i
cross_entropy_total/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
y
cross_entropy_total/ShapeShapeReadout/predicted"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
k
cross_entropy_total/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
{
cross_entropy_total/Shape_1ShapeReadout/predicted"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
j
cross_entropy_total/Sub/yConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
cross_entropy_total/SubSubcross_entropy_total/Rank_1cross_entropy_total/Sub/y"/device:GPU:0*
_output_shapes
: *
T0
�
cross_entropy_total/Slice/beginPackcross_entropy_total/Sub"/device:GPU:0*
T0*

axis *
N*
_output_shapes
:
w
cross_entropy_total/Slice/sizeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
cross_entropy_total/SliceSlicecross_entropy_total/Shape_1cross_entropy_total/Slice/begincross_entropy_total/Slice/size"/device:GPU:0*
Index0*
T0*
_output_shapes
:
�
#cross_entropy_total/concat/values_0Const"/device:GPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
p
cross_entropy_total/concat/axisConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
cross_entropy_total/concatConcatV2#cross_entropy_total/concat/values_0cross_entropy_total/Slicecross_entropy_total/concat/axis"/device:GPU:0*
_output_shapes
:*

Tidx0*
T0*
N
�
cross_entropy_total/ReshapeReshapeReadout/predictedcross_entropy_total/concat"/device:GPU:0*0
_output_shapes
:������������������*
T0*
Tshape0
k
cross_entropy_total/Rank_2Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
~
cross_entropy_total/Shape_2Shapeinput/correct_labels"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
l
cross_entropy_total/Sub_1/yConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
cross_entropy_total/Sub_1Subcross_entropy_total/Rank_2cross_entropy_total/Sub_1/y"/device:GPU:0*
_output_shapes
: *
T0
�
!cross_entropy_total/Slice_1/beginPackcross_entropy_total/Sub_1"/device:GPU:0*
_output_shapes
:*
T0*

axis *
N
y
 cross_entropy_total/Slice_1/sizeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
cross_entropy_total/Slice_1Slicecross_entropy_total/Shape_2!cross_entropy_total/Slice_1/begin cross_entropy_total/Slice_1/size"/device:GPU:0*
_output_shapes
:*
Index0*
T0
�
%cross_entropy_total/concat_1/values_0Const"/device:GPU:0*
valueB:
���������*
dtype0*
_output_shapes
:
r
!cross_entropy_total/concat_1/axisConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
cross_entropy_total/concat_1ConcatV2%cross_entropy_total/concat_1/values_0cross_entropy_total/Slice_1!cross_entropy_total/concat_1/axis"/device:GPU:0*
T0*
N*
_output_shapes
:*

Tidx0
�
cross_entropy_total/Reshape_1Reshapeinput/correct_labelscross_entropy_total/concat_1"/device:GPU:0*
T0*
Tshape0*0
_output_shapes
:������������������
�
1cross_entropy_total/SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitscross_entropy_total/Reshapecross_entropy_total/Reshape_1"/device:GPU:0*
T0*?
_output_shapes-
+:���������:������������������
l
cross_entropy_total/Sub_2/yConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
cross_entropy_total/Sub_2Subcross_entropy_total/Rankcross_entropy_total/Sub_2/y"/device:GPU:0*
_output_shapes
: *
T0
z
!cross_entropy_total/Slice_2/beginConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
 cross_entropy_total/Slice_2/sizePackcross_entropy_total/Sub_2"/device:GPU:0*
T0*

axis *
N*
_output_shapes
:
�
cross_entropy_total/Slice_2Slicecross_entropy_total/Shape!cross_entropy_total/Slice_2/begin cross_entropy_total/Slice_2/size"/device:GPU:0*#
_output_shapes
:���������*
Index0*
T0
�
cross_entropy_total/Reshape_2Reshape1cross_entropy_total/SoftmaxCrossEntropyWithLogitscross_entropy_total/Slice_2"/device:GPU:0*#
_output_shapes
:���������*
T0*
Tshape0
r
cross_entropy_total/ConstConst"/device:GPU:0*
_output_shapes
:*
valueB: *
dtype0
�
cross_entropy_total/MeanMeancross_entropy_total/Reshape_2cross_entropy_total/Const"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
train/gradients/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
train/gradients/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *  �?*
dtype0
z
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const"/device:GPU:0*
T0*
_output_shapes
: 
�
;train/gradients/cross_entropy_total/Mean_grad/Reshape/shapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/cross_entropy_total/Mean_grad/ReshapeReshapetrain/gradients/Fill;train/gradients/cross_entropy_total/Mean_grad/Reshape/shape"/device:GPU:0*
_output_shapes
:*
T0*
Tshape0
�
3train/gradients/cross_entropy_total/Mean_grad/ShapeShapecross_entropy_total/Reshape_2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
2train/gradients/cross_entropy_total/Mean_grad/TileTile5train/gradients/cross_entropy_total/Mean_grad/Reshape3train/gradients/cross_entropy_total/Mean_grad/Shape"/device:GPU:0*#
_output_shapes
:���������*

Tmultiples0*
T0
�
5train/gradients/cross_entropy_total/Mean_grad/Shape_1Shapecross_entropy_total/Reshape_2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
5train/gradients/cross_entropy_total/Mean_grad/Shape_2Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
3train/gradients/cross_entropy_total/Mean_grad/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
2train/gradients/cross_entropy_total/Mean_grad/ProdProd5train/gradients/cross_entropy_total/Mean_grad/Shape_13train/gradients/cross_entropy_total/Mean_grad/Const"/device:GPU:0*
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
5train/gradients/cross_entropy_total/Mean_grad/Const_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
4train/gradients/cross_entropy_total/Mean_grad/Prod_1Prod5train/gradients/cross_entropy_total/Mean_grad/Shape_25train/gradients/cross_entropy_total/Mean_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
_output_shapes
: 
�
7train/gradients/cross_entropy_total/Mean_grad/Maximum/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
dtype0*
_output_shapes
: 
�
5train/gradients/cross_entropy_total/Mean_grad/MaximumMaximum4train/gradients/cross_entropy_total/Mean_grad/Prod_17train/gradients/cross_entropy_total/Mean_grad/Maximum/y"/device:GPU:0*
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
_output_shapes
: 
�
6train/gradients/cross_entropy_total/Mean_grad/floordivFloorDiv2train/gradients/cross_entropy_total/Mean_grad/Prod5train/gradients/cross_entropy_total/Mean_grad/Maximum"/device:GPU:0*
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
_output_shapes
: 
�
2train/gradients/cross_entropy_total/Mean_grad/CastCast6train/gradients/cross_entropy_total/Mean_grad/floordiv"/device:GPU:0*
_output_shapes
: *

DstT0*

SrcT0
�
5train/gradients/cross_entropy_total/Mean_grad/truedivRealDiv2train/gradients/cross_entropy_total/Mean_grad/Tile2train/gradients/cross_entropy_total/Mean_grad/Cast"/device:GPU:0*
T0*#
_output_shapes
:���������
�
8train/gradients/cross_entropy_total/Reshape_2_grad/ShapeShape1cross_entropy_total/SoftmaxCrossEntropyWithLogits)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
:train/gradients/cross_entropy_total/Reshape_2_grad/ReshapeReshape5train/gradients/cross_entropy_total/Mean_grad/truediv8train/gradients/cross_entropy_total/Reshape_2_grad/Shape"/device:GPU:0*
T0*
Tshape0*#
_output_shapes
:���������
�
train/gradients/zeros_like	ZerosLike3cross_entropy_total/SoftmaxCrossEntropyWithLogits:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*0
_output_shapes
:������������������
�
Utrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB :
���������*
dtype0*
_output_shapes
: 
�
Qtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims:train/gradients/cross_entropy_total/Reshape_2_grad/ReshapeUtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim"/device:GPU:0*

Tdim0*
T0*'
_output_shapes
:���������
�
Jtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/mulMulQtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/ExpandDims3cross_entropy_total/SoftmaxCrossEntropyWithLogits:1"/device:GPU:0*0
_output_shapes
:������������������*
T0
�
6train/gradients/cross_entropy_total/Reshape_grad/ShapeShapeReadout/predicted)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
8train/gradients/cross_entropy_total/Reshape_grad/ReshapeReshapeJtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/mul6train/gradients/cross_entropy_total/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0*'
_output_shapes
:���������
�
,train/gradients/Readout/predicted_grad/ShapeShapeReadout/MatMul)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
.train/gradients/Readout/predicted_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:*
dtype0
�
<train/gradients/Readout/predicted_grad/BroadcastGradientArgsBroadcastGradientArgs,train/gradients/Readout/predicted_grad/Shape.train/gradients/Readout/predicted_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
*train/gradients/Readout/predicted_grad/SumSum8train/gradients/cross_entropy_total/Reshape_grad/Reshape<train/gradients/Readout/predicted_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
.train/gradients/Readout/predicted_grad/ReshapeReshape*train/gradients/Readout/predicted_grad/Sum,train/gradients/Readout/predicted_grad/Shape"/device:GPU:0*
T0*
Tshape0*'
_output_shapes
:���������
�
,train/gradients/Readout/predicted_grad/Sum_1Sum8train/gradients/cross_entropy_total/Reshape_grad/Reshape>train/gradients/Readout/predicted_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
0train/gradients/Readout/predicted_grad/Reshape_1Reshape,train/gradients/Readout/predicted_grad/Sum_1.train/gradients/Readout/predicted_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:
�
7train/gradients/Readout/predicted_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1/^train/gradients/Readout/predicted_grad/Reshape1^train/gradients/Readout/predicted_grad/Reshape_1"/device:GPU:0
�
?train/gradients/Readout/predicted_grad/tuple/control_dependencyIdentity.train/gradients/Readout/predicted_grad/Reshape8^train/gradients/Readout/predicted_grad/tuple/group_deps"/device:GPU:0*
T0*A
_class7
53loc:@train/gradients/Readout/predicted_grad/Reshape*'
_output_shapes
:���������
�
Atrain/gradients/Readout/predicted_grad/tuple/control_dependency_1Identity0train/gradients/Readout/predicted_grad/Reshape_18^train/gradients/Readout/predicted_grad/tuple/group_deps"/device:GPU:0*
T0*C
_class9
75loc:@train/gradients/Readout/predicted_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/Readout/MatMul_grad/MatMulMatMul?train/gradients/Readout/predicted_grad/tuple/control_dependencyReadout/Variable/read"/device:GPU:0*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
,train/gradients/Readout/MatMul_grad/MatMul_1MatMul'FC1/batch_normalization/batchnorm/add_1?train/gradients/Readout/predicted_grad/tuple/control_dependency"/device:GPU:0*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
�
4train/gradients/Readout/MatMul_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1+^train/gradients/Readout/MatMul_grad/MatMul-^train/gradients/Readout/MatMul_grad/MatMul_1"/device:GPU:0
�
<train/gradients/Readout/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/Readout/MatMul_grad/MatMul5^train/gradients/Readout/MatMul_grad/tuple/group_deps"/device:GPU:0*=
_class3
1/loc:@train/gradients/Readout/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
>train/gradients/Readout/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/Readout/MatMul_grad/MatMul_15^train/gradients/Readout/MatMul_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:	�*
T0*?
_class5
31loc:@train/gradients/Readout/MatMul_grad/MatMul_1
�
Btrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ShapeShape'FC1/batch_normalization/batchnorm/mul_1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:�*
dtype0
�
Rtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsBtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ShapeDtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/SumSum<train/gradients/Readout/MatMul_grad/tuple/control_dependencyRtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ReshapeReshape@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/SumBtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape"/device:GPU:0*
T0*
Tshape0*(
_output_shapes
:����������
�
Btrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Sum_1Sum<train/gradients/Readout/MatMul_grad/tuple/control_dependencyTtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Ftrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeBtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Sum_1Dtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape_1"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
Mtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1E^train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ReshapeG^train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape_1"/device:GPU:0
�
Utrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityDtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ReshapeN^train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/group_deps"/device:GPU:0*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape
�
Wtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityFtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape_1N^train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/group_deps"/device:GPU:0*
T0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ShapeShapeFC1/Relu)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Rtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsBtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ShapeDtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mulMulUtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency%FC1/batch_normalization/batchnorm/mul"/device:GPU:0*
T0*(
_output_shapes
:����������
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/SumSum@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mulRtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ReshapeReshape@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/SumBtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Shape"/device:GPU:0*
T0*
Tshape0*(
_output_shapes
:����������
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mul_1MulFC1/ReluUtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency"/device:GPU:0*(
_output_shapes
:����������*
T0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Sum_1SumBtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mul_1Ttrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Ftrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeBtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Sum_1Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Shape_1"/device:GPU:0*
Tshape0*
_output_shapes	
:�*
T0
�
Mtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1E^train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ReshapeG^train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1"/device:GPU:0
�
Utrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityDtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ReshapeN^train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps"/device:GPU:0*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
Wtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityFtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1N^train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps"/device:GPU:0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Btrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:�*
dtype0
�
Ptrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/ShapeBtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/SumSumWtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Ptrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgs"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Btrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/ReshapeReshape>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Sum@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Sum_1SumWtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Rtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/NegNeg@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Sum_1"/device:GPU:0*
_output_shapes
:*
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1Reshape>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/NegBtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
Ktrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/ReshapeE^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1"/device:GPU:0
�
Strain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityBtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/ReshapeL^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/group_deps"/device:GPU:0*
T0*U
_classK
IGloc:@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape*
_output_shapes	
:�
�
Utrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityDtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1L^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/group_deps"/device:GPU:0*
_output_shapes	
:�*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:�*
dtype0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:�*
dtype0
�
Rtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsBtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ShapeDtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/mulMulUtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1%FC1/batch_normalization/batchnorm/mul"/device:GPU:0*
_output_shapes	
:�*
T0
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/SumSum@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/mulRtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ReshapeReshape@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/SumBtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/mul_1MulFC1/batch_normalization/SqueezeUtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1"/device:GPU:0*
T0*
_output_shapes	
:�
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Sum_1SumBtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/mul_1Ttrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Ftrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape_1ReshapeBtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Sum_1Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
Mtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1E^train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ReshapeG^train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape_1"/device:GPU:0
�
Utrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityDtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ReshapeN^train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps"/device:GPU:0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape*
_output_shapes	
:�*
T0
�
Wtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityFtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape_1N^train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps"/device:GPU:0*
T0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape_1*
_output_shapes	
:�
�
:train/gradients/FC1/batch_normalization/Squeeze_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
�
<train/gradients/FC1/batch_normalization/Squeeze_grad/ReshapeReshapeUtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency:train/gradients/FC1/batch_normalization/Squeeze_grad/Shape"/device:GPU:0*
_output_shapes
:	�*
T0*
Tshape0
�
train/gradients/AddNAddNWtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1Wtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1"/device:GPU:0*
T0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Ptrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/ShapeBtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
>train/gradients/FC1/batch_normalization/batchnorm/mul_grad/mulMultrain/gradients/AddN batch_normalization_3/gamma/read"/device:GPU:0*
_output_shapes	
:�*
T0
�
>train/gradients/FC1/batch_normalization/batchnorm/mul_grad/SumSum>train/gradients/FC1/batch_normalization/batchnorm/mul_grad/mulPtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/ReshapeReshape>train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Sum@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/mul_1Mul'FC1/batch_normalization/batchnorm/Rsqrttrain/gradients/AddN"/device:GPU:0*
_output_shapes	
:�*
T0
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Sum_1Sum@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/mul_1Rtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1Reshape@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Sum_1Btrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
Ktrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/FC1/batch_normalization/batchnorm/mul_grad/ReshapeE^train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1"/device:GPU:0
�
Strain/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityBtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/ReshapeL^train/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/group_deps"/device:GPU:0*
T0*U
_classK
IGloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape*
_output_shapes	
:�
�
Utrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityDtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1L^train/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/group_deps"/device:GPU:0*
_output_shapes	
:�*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1
�
>train/gradients/FC1/batch_normalization/Select_grad/zeros_likeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:	�*
valueB	�*    *
dtype0
�
:train/gradients/FC1/batch_normalization/Select_grad/SelectSelectFC1/batch_normalization/Reshape<train/gradients/FC1/batch_normalization/Squeeze_grad/Reshape>train/gradients/FC1/batch_normalization/Select_grad/zeros_like"/device:GPU:0*
_output_shapes
:	�*
T0
�
<train/gradients/FC1/batch_normalization/Select_grad/Select_1SelectFC1/batch_normalization/Reshape>train/gradients/FC1/batch_normalization/Select_grad/zeros_like<train/gradients/FC1/batch_normalization/Squeeze_grad/Reshape"/device:GPU:0*
T0*
_output_shapes
:	�
�
Dtrain/gradients/FC1/batch_normalization/Select_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1;^train/gradients/FC1/batch_normalization/Select_grad/Select=^train/gradients/FC1/batch_normalization/Select_grad/Select_1"/device:GPU:0
�
Ltrain/gradients/FC1/batch_normalization/Select_grad/tuple/control_dependencyIdentity:train/gradients/FC1/batch_normalization/Select_grad/SelectE^train/gradients/FC1/batch_normalization/Select_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:	�*
T0*M
_classC
A?loc:@train/gradients/FC1/batch_normalization/Select_grad/Select
�
Ntrain/gradients/FC1/batch_normalization/Select_grad/tuple/control_dependency_1Identity<train/gradients/FC1/batch_normalization/Select_grad/Select_1E^train/gradients/FC1/batch_normalization/Select_grad/tuple/group_deps"/device:GPU:0*
T0*O
_classE
CAloc:@train/gradients/FC1/batch_normalization/Select_grad/Select_1*
_output_shapes
:	�
�
Ftrain/gradients/FC1/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGrad	RsqrtGrad'FC1/batch_normalization/batchnorm/RsqrtStrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/control_dependency"/device:GPU:0*
_output_shapes	
:�*
T0
�
=train/gradients/FC1/batch_normalization/ExpandDims_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
?train/gradients/FC1/batch_normalization/ExpandDims_grad/ReshapeReshapeLtrain/gradients/FC1/batch_normalization/Select_grad/tuple/control_dependency=train/gradients/FC1/batch_normalization/ExpandDims_grad/Shape"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
@train/gradients/FC1/batch_normalization/batchnorm/add_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Btrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
Ptrain/gradients/FC1/batch_normalization/batchnorm/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/FC1/batch_normalization/batchnorm/add_grad/ShapeBtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
>train/gradients/FC1/batch_normalization/batchnorm/add_grad/SumSumFtrain/gradients/FC1/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGradPtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/add_grad/ReshapeReshape>train/gradients/FC1/batch_normalization/batchnorm/add_grad/Sum@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Shape"/device:GPU:0*
Tshape0*
_output_shapes	
:�*
T0
�
@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Sum_1SumFtrain/gradients/FC1/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGradRtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Reshape_1Reshape@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Sum_1Btrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes
: 
�
Ktrain/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/FC1/batch_normalization/batchnorm/add_grad/ReshapeE^train/gradients/FC1/batch_normalization/batchnorm/add_grad/Reshape_1"/device:GPU:0
�
Strain/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/control_dependencyIdentityBtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/ReshapeL^train/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/group_deps"/device:GPU:0*U
_classK
IGloc:@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Reshape*
_output_shapes	
:�*
T0
�
Utrain/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Reshape_1L^train/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
: *
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Reshape_1
�
Btrain/gradients/FC1/batch_normalization/moments/Squeeze_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0
�
Dtrain/gradients/FC1/batch_normalization/moments/Squeeze_grad/ReshapeReshape?train/gradients/FC1/batch_normalization/ExpandDims_grad/ReshapeBtrain/gradients/FC1/batch_normalization/moments/Squeeze_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:	�
�
<train/gradients/FC1/batch_normalization/Squeeze_1_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0
�
>train/gradients/FC1/batch_normalization/Squeeze_1_grad/ReshapeReshapeStrain/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/control_dependency<train/gradients/FC1/batch_normalization/Squeeze_1_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:	�
�
@train/gradients/FC1/batch_normalization/Select_1_grad/zeros_likeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
<train/gradients/FC1/batch_normalization/Select_1_grad/SelectSelect!FC1/batch_normalization/Reshape_1>train/gradients/FC1/batch_normalization/Squeeze_1_grad/Reshape@train/gradients/FC1/batch_normalization/Select_1_grad/zeros_like"/device:GPU:0*
_output_shapes
:	�*
T0
�
>train/gradients/FC1/batch_normalization/Select_1_grad/Select_1Select!FC1/batch_normalization/Reshape_1@train/gradients/FC1/batch_normalization/Select_1_grad/zeros_like>train/gradients/FC1/batch_normalization/Squeeze_1_grad/Reshape"/device:GPU:0*
T0*
_output_shapes
:	�
�
Ftrain/gradients/FC1/batch_normalization/Select_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1=^train/gradients/FC1/batch_normalization/Select_1_grad/Select?^train/gradients/FC1/batch_normalization/Select_1_grad/Select_1"/device:GPU:0
�
Ntrain/gradients/FC1/batch_normalization/Select_1_grad/tuple/control_dependencyIdentity<train/gradients/FC1/batch_normalization/Select_1_grad/SelectG^train/gradients/FC1/batch_normalization/Select_1_grad/tuple/group_deps"/device:GPU:0*
T0*O
_classE
CAloc:@train/gradients/FC1/batch_normalization/Select_1_grad/Select*
_output_shapes
:	�
�
Ptrain/gradients/FC1/batch_normalization/Select_1_grad/tuple/control_dependency_1Identity>train/gradients/FC1/batch_normalization/Select_1_grad/Select_1G^train/gradients/FC1/batch_normalization/Select_1_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:	�*
T0*Q
_classG
ECloc:@train/gradients/FC1/batch_normalization/Select_1_grad/Select_1
�
?train/gradients/FC1/batch_normalization/ExpandDims_2_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Atrain/gradients/FC1/batch_normalization/ExpandDims_2_grad/ReshapeReshapeNtrain/gradients/FC1/batch_normalization/Select_1_grad/tuple/control_dependency?train/gradients/FC1/batch_normalization/ExpandDims_2_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
Dtrain/gradients/FC1/batch_normalization/moments/Squeeze_1_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
�
Ftrain/gradients/FC1/batch_normalization/moments/Squeeze_1_grad/ReshapeReshapeAtrain/gradients/FC1/batch_normalization/ExpandDims_2_grad/ReshapeDtrain/gradients/FC1/batch_normalization/moments/Squeeze_1_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:	�
�
Ctrain/gradients/FC1/batch_normalization/moments/variance_grad/ShapeShape1FC1/batch_normalization/moments/SquaredDifference)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/SizeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Atrain/gradients/FC1/batch_normalization/moments/variance_grad/addAdd:FC1/batch_normalization/moments/variance/reduction_indicesBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Size"/device:GPU:0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
_output_shapes
:*
T0
�
Atrain/gradients/FC1/batch_normalization/moments/variance_grad/modFloorModAtrain/gradients/FC1/batch_normalization/moments/variance_grad/addBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Size"/device:GPU:0*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
_output_shapes
:
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0*
_output_shapes
:
�
Itrain/gradients/FC1/batch_normalization/moments/variance_grad/range/startConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B : *V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Itrain/gradients/FC1/batch_normalization/moments/variance_grad/range/deltaConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Ctrain/gradients/FC1/batch_normalization/moments/variance_grad/rangeRangeItrain/gradients/FC1/batch_normalization/moments/variance_grad/range/startBtrain/gradients/FC1/batch_normalization/moments/variance_grad/SizeItrain/gradients/FC1/batch_normalization/moments/variance_grad/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape
�
Htrain/gradients/FC1/batch_normalization/moments/variance_grad/Fill/valueConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/FillFillEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_1Htrain/gradients/FC1/batch_normalization/moments/variance_grad/Fill/value"/device:GPU:0*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
_output_shapes
:
�
Ktrain/gradients/FC1/batch_normalization/moments/variance_grad/DynamicStitchDynamicStitchCtrain/gradients/FC1/batch_normalization/moments/variance_grad/rangeAtrain/gradients/FC1/batch_normalization/moments/variance_grad/modCtrain/gradients/FC1/batch_normalization/moments/variance_grad/ShapeBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Fill"/device:GPU:0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
N*#
_output_shapes
:���������*
T0
�
Gtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
value	B :*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/MaximumMaximumKtrain/gradients/FC1/batch_normalization/moments/variance_grad/DynamicStitchGtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum/y"/device:GPU:0*#
_output_shapes
:���������*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape
�
Ftrain/gradients/FC1/batch_normalization/moments/variance_grad/floordivFloorDivCtrain/gradients/FC1/batch_normalization/moments/variance_grad/ShapeEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum"/device:GPU:0*
_output_shapes
:*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/ReshapeReshapeFtrain/gradients/FC1/batch_normalization/moments/Squeeze_1_grad/ReshapeKtrain/gradients/FC1/batch_normalization/moments/variance_grad/DynamicStitch"/device:GPU:0*
Tshape0*
_output_shapes
:*
T0
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/TileTileEtrain/gradients/FC1/batch_normalization/moments/variance_grad/ReshapeFtrain/gradients/FC1/batch_normalization/moments/variance_grad/floordiv"/device:GPU:0*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2Shape1FC1/batch_normalization/moments/SquaredDifference)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_3Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0
�
Ctrain/gradients/FC1/batch_normalization/moments/variance_grad/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/ProdProdEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2Ctrain/gradients/FC1/batch_normalization/moments/variance_grad/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: 
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Const_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
:
�
Dtrain/gradients/FC1/batch_normalization/moments/variance_grad/Prod_1ProdEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_3Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: 
�
Itrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
: 
�
Gtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1MaximumDtrain/gradients/FC1/batch_normalization/moments/variance_grad/Prod_1Itrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1/y"/device:GPU:0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: *
T0
�
Htrain/gradients/FC1/batch_normalization/moments/variance_grad/floordiv_1FloorDivBtrain/gradients/FC1/batch_normalization/moments/variance_grad/ProdGtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1"/device:GPU:0*
T0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: 
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/CastCastHtrain/gradients/FC1/batch_normalization/moments/variance_grad/floordiv_1"/device:GPU:0*

SrcT0*
_output_shapes
: *

DstT0
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/truedivRealDivBtrain/gradients/FC1/batch_normalization/moments/variance_grad/TileBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Cast"/device:GPU:0*(
_output_shapes
:����������*
T0
�
Ltrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ShapeShapeFC1/Relu)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
Ntrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0
�
\train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ShapeNtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
Mtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/scalarConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1F^train/gradients/FC1/batch_normalization/moments/variance_grad/truediv"/device:GPU:0*
_output_shapes
: *
valueB
 *   @*
dtype0
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mulMulMtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/scalarEtrain/gradients/FC1/batch_normalization/moments/variance_grad/truediv"/device:GPU:0*
T0*(
_output_shapes
:����������
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/subSubFC1/Relu,FC1/batch_normalization/moments/StopGradient)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1F^train/gradients/FC1/batch_normalization/moments/variance_grad/truediv"/device:GPU:0*
T0*(
_output_shapes
:����������
�
Ltrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mul_1MulJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mulJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/sub"/device:GPU:0*
T0*(
_output_shapes
:����������
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/SumSumLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mul_1\train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Ntrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ReshapeReshapeJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/SumLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Shape"/device:GPU:0*
T0*
Tshape0*(
_output_shapes
:����������
�
Ltrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Sum_1SumLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mul_1^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Ptrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Reshape_1ReshapeLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Sum_1Ntrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:	�
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/NegNegPtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Reshape_1"/device:GPU:0*
T0*
_output_shapes
:	�
�
Wtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1O^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ReshapeK^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Neg"/device:GPU:0
�
_train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependencyIdentityNtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ReshapeX^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/group_deps"/device:GPU:0*
T0*a
_classW
USloc:@train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Reshape*(
_output_shapes
:����������
�
atrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependency_1IdentityJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/NegX^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:	�*
T0*]
_classS
QOloc:@train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Neg
�
?train/gradients/FC1/batch_normalization/moments/mean_grad/ShapeShapeFC1/Relu)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/SizeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
=train/gradients/FC1/batch_normalization/moments/mean_grad/addAdd6FC1/batch_normalization/moments/mean/reduction_indices>train/gradients/FC1/batch_normalization/moments/mean_grad/Size"/device:GPU:0*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
_output_shapes
:
�
=train/gradients/FC1/batch_normalization/moments/mean_grad/modFloorMod=train/gradients/FC1/batch_normalization/moments/mean_grad/add>train/gradients/FC1/batch_normalization/moments/mean_grad/Size"/device:GPU:0*
_output_shapes
:*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0*
_output_shapes
:
�
Etrain/gradients/FC1/batch_normalization/moments/mean_grad/range/startConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
value	B : *R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0
�
Etrain/gradients/FC1/batch_normalization/moments/mean_grad/range/deltaConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
?train/gradients/FC1/batch_normalization/moments/mean_grad/rangeRangeEtrain/gradients/FC1/batch_normalization/moments/mean_grad/range/start>train/gradients/FC1/batch_normalization/moments/mean_grad/SizeEtrain/gradients/FC1/batch_normalization/moments/mean_grad/range/delta"/device:GPU:0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
_output_shapes
:*

Tidx0
�
Dtrain/gradients/FC1/batch_normalization/moments/mean_grad/Fill/valueConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/FillFillAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_1Dtrain/gradients/FC1/batch_normalization/moments/mean_grad/Fill/value"/device:GPU:0*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
_output_shapes
:
�
Gtrain/gradients/FC1/batch_normalization/moments/mean_grad/DynamicStitchDynamicStitch?train/gradients/FC1/batch_normalization/moments/mean_grad/range=train/gradients/FC1/batch_normalization/moments/mean_grad/mod?train/gradients/FC1/batch_normalization/moments/mean_grad/Shape>train/gradients/FC1/batch_normalization/moments/mean_grad/Fill"/device:GPU:0*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
N*#
_output_shapes
:���������
�
Ctrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/MaximumMaximumGtrain/gradients/FC1/batch_normalization/moments/mean_grad/DynamicStitchCtrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum/y"/device:GPU:0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*#
_output_shapes
:���������*
T0
�
Btrain/gradients/FC1/batch_normalization/moments/mean_grad/floordivFloorDiv?train/gradients/FC1/batch_normalization/moments/mean_grad/ShapeAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum"/device:GPU:0*
_output_shapes
:*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/ReshapeReshapeDtrain/gradients/FC1/batch_normalization/moments/Squeeze_grad/ReshapeGtrain/gradients/FC1/batch_normalization/moments/mean_grad/DynamicStitch"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/TileTileAtrain/gradients/FC1/batch_normalization/moments/mean_grad/ReshapeBtrain/gradients/FC1/batch_normalization/moments/mean_grad/floordiv"/device:GPU:0*

Tmultiples0*
T0*0
_output_shapes
:������������������
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2ShapeFC1/Relu)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_3Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
�
?train/gradients/FC1/batch_normalization/moments/mean_grad/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/ProdProdAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2?train/gradients/FC1/batch_normalization/moments/mean_grad/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
_output_shapes
: 
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Const_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
�
@train/gradients/FC1/batch_normalization/moments/mean_grad/Prod_1ProdAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_3Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Const_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
_output_shapes
: 
�
Etrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum_1/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
: 
�
Ctrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum_1Maximum@train/gradients/FC1/batch_normalization/moments/mean_grad/Prod_1Etrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum_1/y"/device:GPU:0*
T0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
_output_shapes
: 
�
Dtrain/gradients/FC1/batch_normalization/moments/mean_grad/floordiv_1FloorDiv>train/gradients/FC1/batch_normalization/moments/mean_grad/ProdCtrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum_1"/device:GPU:0*
T0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
_output_shapes
: 
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/CastCastDtrain/gradients/FC1/batch_normalization/moments/mean_grad/floordiv_1"/device:GPU:0*
_output_shapes
: *

DstT0*

SrcT0
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/truedivRealDiv>train/gradients/FC1/batch_normalization/moments/mean_grad/Tile>train/gradients/FC1/batch_normalization/moments/mean_grad/Cast"/device:GPU:0*(
_output_shapes
:����������*
T0
�
train/gradients/AddN_1AddNUtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependencyAtrain/gradients/FC1/batch_normalization/moments/mean_grad/truediv"/device:GPU:0*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape*
N*(
_output_shapes
:����������
�
&train/gradients/FC1/Relu_grad/ReluGradReluGradtrain/gradients/AddN_1FC1/Relu"/device:GPU:0*(
_output_shapes
:����������*
T0
�
"train/gradients/FC1/add_grad/ShapeShape
FC1/MatMul)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
$train/gradients/FC1/add_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:�*
dtype0
�
2train/gradients/FC1/add_grad/BroadcastGradientArgsBroadcastGradientArgs"train/gradients/FC1/add_grad/Shape$train/gradients/FC1/add_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
 train/gradients/FC1/add_grad/SumSum&train/gradients/FC1/Relu_grad/ReluGrad2train/gradients/FC1/add_grad/BroadcastGradientArgs"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
$train/gradients/FC1/add_grad/ReshapeReshape train/gradients/FC1/add_grad/Sum"train/gradients/FC1/add_grad/Shape"/device:GPU:0*
T0*
Tshape0*(
_output_shapes
:����������
�
"train/gradients/FC1/add_grad/Sum_1Sum&train/gradients/FC1/Relu_grad/ReluGrad4train/gradients/FC1/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
&train/gradients/FC1/add_grad/Reshape_1Reshape"train/gradients/FC1/add_grad/Sum_1$train/gradients/FC1/add_grad/Shape_1"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
-train/gradients/FC1/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1%^train/gradients/FC1/add_grad/Reshape'^train/gradients/FC1/add_grad/Reshape_1"/device:GPU:0
�
5train/gradients/FC1/add_grad/tuple/control_dependencyIdentity$train/gradients/FC1/add_grad/Reshape.^train/gradients/FC1/add_grad/tuple/group_deps"/device:GPU:0*
T0*7
_class-
+)loc:@train/gradients/FC1/add_grad/Reshape*(
_output_shapes
:����������
�
7train/gradients/FC1/add_grad/tuple/control_dependency_1Identity&train/gradients/FC1/add_grad/Reshape_1.^train/gradients/FC1/add_grad/tuple/group_deps"/device:GPU:0*
T0*9
_class/
-+loc:@train/gradients/FC1/add_grad/Reshape_1*
_output_shapes	
:�
�
&train/gradients/FC1/MatMul_grad/MatMulMatMul5train/gradients/FC1/add_grad/tuple/control_dependencyFC1/Variable/read"/device:GPU:0*
T0*)
_output_shapes
:�����������*
transpose_a( *
transpose_b(
�
(train/gradients/FC1/MatMul_grad/MatMul_1MatMulFC1/Reshape5train/gradients/FC1/add_grad/tuple/control_dependency"/device:GPU:0*
T0*!
_output_shapes
:���*
transpose_a(*
transpose_b( 
�
0train/gradients/FC1/MatMul_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1'^train/gradients/FC1/MatMul_grad/MatMul)^train/gradients/FC1/MatMul_grad/MatMul_1"/device:GPU:0
�
8train/gradients/FC1/MatMul_grad/tuple/control_dependencyIdentity&train/gradients/FC1/MatMul_grad/MatMul1^train/gradients/FC1/MatMul_grad/tuple/group_deps"/device:GPU:0*9
_class/
-+loc:@train/gradients/FC1/MatMul_grad/MatMul*)
_output_shapes
:�����������*
T0
�
:train/gradients/FC1/MatMul_grad/tuple/control_dependency_1Identity(train/gradients/FC1/MatMul_grad/MatMul_11^train/gradients/FC1/MatMul_grad/tuple/group_deps"/device:GPU:0*!
_output_shapes
:���*
T0*;
_class1
/-loc:@train/gradients/FC1/MatMul_grad/MatMul_1
�
&train/gradients/FC1/Reshape_grad/ShapeShape#CNN3/batch_normalization/cond/Merge)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
(train/gradients/FC1/Reshape_grad/ReshapeReshape8train/gradients/FC1/MatMul_grad/tuple/control_dependency&train/gradients/FC1/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0*/
_output_shapes
:���������@
�
Btrain/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_gradSwitch(train/gradients/FC1/Reshape_grad/Reshape%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*;
_class1
/-loc:@train/gradients/FC1/Reshape_grad/Reshape*J
_output_shapes8
6:���������@:���������@
�
Itrain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_grad"/device:GPU:0
�
Qtrain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentityBtrain/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_gradJ^train/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*
T0*;
_class1
/-loc:@train/gradients/FC1/Reshape_grad/Reshape*/
_output_shapes
:���������@
�
Strain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/control_dependency_1IdentityDtrain/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_grad:1J^train/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������@*
T0*;
_class1
/-loc:@train/gradients/FC1/Reshape_grad/Reshape
�
train/gradients/zeros_like_1	ZerosLike0CNN3/batch_normalization/cond/FusedBatchNorm_1:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:
�
train/gradients/zeros_like_2	ZerosLike0CNN3/batch_normalization/cond/FusedBatchNorm_1:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0
�
train/gradients/zeros_like_3	ZerosLike0CNN3/batch_normalization/cond/FusedBatchNorm_1:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0
�
train/gradients/zeros_like_4	ZerosLike0CNN3/batch_normalization/cond/FusedBatchNorm_1:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:
�
Rtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Mtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose	Transpose5CNN3/batch_normalization/cond/FusedBatchNorm_1/SwitchRtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/perm"/device:GPU:0*
T0*/
_output_shapes
:���������@*
Tperm0
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Otrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1	TransposeQtrain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/control_dependencyTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/perm"/device:GPU:0*/
_output_shapes
:���������@*
Tperm0*
T0
�
Vtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradOtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1Mtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
T0*
data_formatNHWC*G
_output_shapes5
3:���������@::::*
is_training( *
epsilon%o�:
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Otrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2	TransposeVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/perm"/device:GPU:0*
T0*/
_output_shapes
:���������@*
Tperm0
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1W^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradP^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2"/device:GPU:0
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityOtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*b
_classX
VTloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2*/
_output_shapes
:���������@
�
^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityXtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
�
^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityXtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
train/gradients/zeros_like_5	ZerosLike.CNN3/batch_normalization/cond/FusedBatchNorm:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0
�
train/gradients/zeros_like_6	ZerosLike.CNN3/batch_normalization/cond/FusedBatchNorm:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0
�
train/gradients/zeros_like_7	ZerosLike.CNN3/batch_normalization/cond/FusedBatchNorm:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0
�
train/gradients/zeros_like_8	ZerosLike.CNN3/batch_normalization/cond/FusedBatchNorm:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradStrain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/control_dependency_15CNN3/batch_normalization/cond/FusedBatchNorm/Switch:17CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1:1.CNN3/batch_normalization/cond/FusedBatchNorm:3.CNN3/batch_normalization/cond/FusedBatchNorm:4"/device:GPU:0*C
_output_shapes1
/:���������@::: : *
is_training(*
epsilon%o�:*
T0*
data_formatNCHW
�
Rtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad"/device:GPU:0
�
Ztrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradS^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������@*
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
train/gradients/SwitchSwitch	CNN3/Relu%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*J
_output_shapes8
6:���������@:���������@*
T0
~
train/gradients/Shape_1Shapetrain/gradients/Switch:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zerosFilltrain/gradients/Shape_1train/gradients/zeros/Const"/device:GPU:0*
T0*/
_output_shapes
:���������@
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMerge\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencytrain/gradients/zeros"/device:GPU:0*
T0*
N*1
_output_shapes
:���������@: 
�
train/gradients/Switch_1Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
::
�
train/gradients/Shape_2Shapetrain/gradients/Switch_1:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_1/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_1Filltrain/gradients/Shape_2train/gradients/zeros_1/Const"/device:GPU:0*
T0*
_output_shapes
:
�
Vtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMerge^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1train/gradients/zeros_1"/device:GPU:0*
T0*
N*
_output_shapes

:: 
�
train/gradients/Switch_2Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
::
�
train/gradients/Shape_3Shapetrain/gradients/Switch_2:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_2/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_2Filltrain/gradients/Shape_3train/gradients/zeros_2/Const"/device:GPU:0*
T0*
_output_shapes
:
�
Vtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMerge^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2train/gradients/zeros_2"/device:GPU:0*
T0*
N*
_output_shapes

:: 
�
train/gradients/Switch_3Switch	CNN3/Relu%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*J
_output_shapes8
6:���������@:���������@
~
train/gradients/Shape_4Shapetrain/gradients/Switch_3"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
train/gradients/zeros_3/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_3Filltrain/gradients/Shape_4train/gradients/zeros_3/Const"/device:GPU:0*/
_output_shapes
:���������@*
T0
�
Rtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergeZtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencytrain/gradients/zeros_3"/device:GPU:0*
T0*
N*1
_output_shapes
:���������@: 
�
train/gradients/Switch_4Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
::
~
train/gradients/Shape_5Shapetrain/gradients/Switch_4"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
train/gradients/zeros_4/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_4Filltrain/gradients/Shape_5train/gradients/zeros_4/Const"/device:GPU:0*
T0*
_output_shapes
:
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMerge\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1train/gradients/zeros_4"/device:GPU:0*
N*
_output_shapes

:: *
T0
�
train/gradients/Switch_5Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
::
~
train/gradients/Shape_6Shapetrain/gradients/Switch_5"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_5/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_5Filltrain/gradients/Shape_6train/gradients/zeros_5/Const"/device:GPU:0*
T0*
_output_shapes
:
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMerge\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2train/gradients/zeros_5"/device:GPU:0*
N*
_output_shapes

:: *
T0
�
train/gradients/AddN_2AddNTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradRtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������@*
T0
�
'train/gradients/CNN3/Relu_grad/ReluGradReluGradtrain/gradients/AddN_2	CNN3/Relu"/device:GPU:0*
T0*/
_output_shapes
:���������@
�
train/gradients/AddN_3AddNVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad"/device:GPU:0*
_output_shapes
:*
T0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N
�
train/gradients/AddN_4AddNVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
#train/gradients/CNN3/add_grad/ShapeShapeCNN3/Conv2D)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
%train/gradients/CNN3/add_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:@*
dtype0*
_output_shapes
:
�
3train/gradients/CNN3/add_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/CNN3/add_grad/Shape%train/gradients/CNN3/add_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
!train/gradients/CNN3/add_grad/SumSum'train/gradients/CNN3/Relu_grad/ReluGrad3train/gradients/CNN3/add_grad/BroadcastGradientArgs"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
%train/gradients/CNN3/add_grad/ReshapeReshape!train/gradients/CNN3/add_grad/Sum#train/gradients/CNN3/add_grad/Shape"/device:GPU:0*
T0*
Tshape0*/
_output_shapes
:���������@
�
#train/gradients/CNN3/add_grad/Sum_1Sum'train/gradients/CNN3/Relu_grad/ReluGrad5train/gradients/CNN3/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
'train/gradients/CNN3/add_grad/Reshape_1Reshape#train/gradients/CNN3/add_grad/Sum_1%train/gradients/CNN3/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:@
�
.train/gradients/CNN3/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1&^train/gradients/CNN3/add_grad/Reshape(^train/gradients/CNN3/add_grad/Reshape_1"/device:GPU:0
�
6train/gradients/CNN3/add_grad/tuple/control_dependencyIdentity%train/gradients/CNN3/add_grad/Reshape/^train/gradients/CNN3/add_grad/tuple/group_deps"/device:GPU:0*8
_class.
,*loc:@train/gradients/CNN3/add_grad/Reshape*/
_output_shapes
:���������@*
T0
�
8train/gradients/CNN3/add_grad/tuple/control_dependency_1Identity'train/gradients/CNN3/add_grad/Reshape_1/^train/gradients/CNN3/add_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:@*
T0*:
_class0
.,loc:@train/gradients/CNN3/add_grad/Reshape_1
�
'train/gradients/CNN3/Conv2D_grad/ShapeNShapeN#CNN2/batch_normalization/cond/MergeCNN3/weights/Variable/read)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
::*
T0*
out_type0*
N
�
4train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'train/gradients/CNN3/Conv2D_grad/ShapeNCNN3/weights/Variable/read6train/gradients/CNN3/add_grad/tuple/control_dependency"/device:GPU:0*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC
�
5train/gradients/CNN3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter#CNN2/batch_normalization/cond/Merge)train/gradients/CNN3/Conv2D_grad/ShapeN:16train/gradients/CNN3/add_grad/tuple/control_dependency"/device:GPU:0*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
1train/gradients/CNN3/Conv2D_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_15^train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput6^train/gradients/CNN3/Conv2D_grad/Conv2DBackpropFilter"/device:GPU:0
�
9train/gradients/CNN3/Conv2D_grad/tuple/control_dependencyIdentity4train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput2^train/gradients/CNN3/Conv2D_grad/tuple/group_deps"/device:GPU:0*
T0*G
_class=
;9loc:@train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������   
�
;train/gradients/CNN3/Conv2D_grad/tuple/control_dependency_1Identity5train/gradients/CNN3/Conv2D_grad/Conv2DBackpropFilter2^train/gradients/CNN3/Conv2D_grad/tuple/group_deps"/device:GPU:0*&
_output_shapes
: @*
T0*H
_class>
<:loc:@train/gradients/CNN3/Conv2D_grad/Conv2DBackpropFilter
�
Btrain/gradients/CNN2/batch_normalization/cond/Merge_grad/cond_gradSwitch9train/gradients/CNN3/Conv2D_grad/tuple/control_dependency%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*J
_output_shapes8
6:���������   :���������   *
T0*G
_class=
;9loc:@train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput
�
Itrain/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/CNN2/batch_normalization/cond/Merge_grad/cond_grad"/device:GPU:0
�
Qtrain/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentityBtrain/gradients/CNN2/batch_normalization/cond/Merge_grad/cond_gradJ^train/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������   *
T0*G
_class=
;9loc:@train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput
�
Strain/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/control_dependency_1IdentityDtrain/gradients/CNN2/batch_normalization/cond/Merge_grad/cond_grad:1J^train/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*
T0*G
_class=
;9loc:@train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������   
�
train/gradients/zeros_like_9	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
T0
�
train/gradients/zeros_like_10	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
T0
�
train/gradients/zeros_like_11	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
train/gradients/zeros_like_12	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
T0
�
Rtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*%
valueB"             *
dtype0
�
Mtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose	Transpose5CNN2/batch_normalization/cond/FusedBatchNorm_1/SwitchRtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/perm"/device:GPU:0*/
_output_shapes
:���������   *
Tperm0*
T0
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Otrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1	TransposeQtrain/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/control_dependencyTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/perm"/device:GPU:0*/
_output_shapes
:���������   *
Tperm0*
T0
�
Vtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradOtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1Mtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
T0*
data_formatNHWC*G
_output_shapes5
3:���������   : : : : *
is_training( *
epsilon%o�:
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Otrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2	TransposeVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/perm"/device:GPU:0*/
_output_shapes
:���������   *
Tperm0*
T0
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1W^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradP^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2"/device:GPU:0
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityOtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*b
_classX
VTloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2*/
_output_shapes
:���������   *
T0
�
^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityXtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
: 
�
^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityXtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
train/gradients/zeros_like_13	ZerosLike.CNN2/batch_normalization/cond/FusedBatchNorm:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
train/gradients/zeros_like_14	ZerosLike.CNN2/batch_normalization/cond/FusedBatchNorm:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
train/gradients/zeros_like_15	ZerosLike.CNN2/batch_normalization/cond/FusedBatchNorm:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
train/gradients/zeros_like_16	ZerosLike.CNN2/batch_normalization/cond/FusedBatchNorm:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradStrain/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/control_dependency_15CNN2/batch_normalization/cond/FusedBatchNorm/Switch:17CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1:1.CNN2/batch_normalization/cond/FusedBatchNorm:3.CNN2/batch_normalization/cond/FusedBatchNorm:4"/device:GPU:0*
T0*
data_formatNCHW*C
_output_shapes1
/:���������   : : : : *
is_training(*
epsilon%o�:
�
Rtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad"/device:GPU:0
�
Ztrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradS^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������   
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
train/gradients/Switch_6Switch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*J
_output_shapes8
6:���������   :���������   *
T0
�
train/gradients/Shape_7Shapetrain/gradients/Switch_6:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_6/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_6Filltrain/gradients/Shape_7train/gradients/zeros_6/Const"/device:GPU:0*/
_output_shapes
:���������   *
T0
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMerge\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencytrain/gradients/zeros_6"/device:GPU:0*
N*1
_output_shapes
:���������   : *
T0
�
train/gradients/Switch_7Switch batch_normalization_1/gamma/read%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
: : 
�
train/gradients/Shape_8Shapetrain/gradients/Switch_7:1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
train/gradients/zeros_7/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_7Filltrain/gradients/Shape_8train/gradients/zeros_7/Const"/device:GPU:0*
T0*
_output_shapes
: 
�
Vtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMerge^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1train/gradients/zeros_7"/device:GPU:0*
_output_shapes

: : *
T0*
N
�
train/gradients/Switch_8Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
: : *
T0
�
train/gradients/Shape_9Shapetrain/gradients/Switch_8:1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
train/gradients/zeros_8/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_8Filltrain/gradients/Shape_9train/gradients/zeros_8/Const"/device:GPU:0*
T0*
_output_shapes
: 
�
Vtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMerge^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2train/gradients/zeros_8"/device:GPU:0*
T0*
N*
_output_shapes

: : 
�
train/gradients/Switch_9Switch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*J
_output_shapes8
6:���������   :���������   *
T0

train/gradients/Shape_10Shapetrain/gradients/Switch_9"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_9/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_9Filltrain/gradients/Shape_10train/gradients/zeros_9/Const"/device:GPU:0*
T0*/
_output_shapes
:���������   
�
Rtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergeZtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencytrain/gradients/zeros_9"/device:GPU:0*
T0*
N*1
_output_shapes
:���������   : 
�
train/gradients/Switch_10Switch batch_normalization_1/gamma/read%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
: : *
T0
�
train/gradients/Shape_11Shapetrain/gradients/Switch_10"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_10/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_10Filltrain/gradients/Shape_11train/gradients/zeros_10/Const"/device:GPU:0*
T0*
_output_shapes
: 
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMerge\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1train/gradients/zeros_10"/device:GPU:0*
T0*
N*
_output_shapes

: : 
�
train/gradients/Switch_11Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
: : *
T0
�
train/gradients/Shape_12Shapetrain/gradients/Switch_11"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
train/gradients/zeros_11/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_11Filltrain/gradients/Shape_12train/gradients/zeros_11/Const"/device:GPU:0*
T0*
_output_shapes
: 
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMerge\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2train/gradients/zeros_11"/device:GPU:0*
T0*
N*
_output_shapes

: : 
�
train/gradients/AddN_5AddNTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradRtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������   
�
'train/gradients/CNN2/Relu_grad/ReluGradReluGradtrain/gradients/AddN_5	CNN2/Relu"/device:GPU:0*/
_output_shapes
:���������   *
T0
�
train/gradients/AddN_6AddNVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
: 
�
train/gradients/AddN_7AddNVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad"/device:GPU:0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
: *
T0
�
#train/gradients/CNN2/add_grad/ShapeShapeCNN2/Conv2D)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
%train/gradients/CNN2/add_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
3train/gradients/CNN2/add_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/CNN2/add_grad/Shape%train/gradients/CNN2/add_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
!train/gradients/CNN2/add_grad/SumSum'train/gradients/CNN2/Relu_grad/ReluGrad3train/gradients/CNN2/add_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
%train/gradients/CNN2/add_grad/ReshapeReshape!train/gradients/CNN2/add_grad/Sum#train/gradients/CNN2/add_grad/Shape"/device:GPU:0*
T0*
Tshape0*/
_output_shapes
:���������   
�
#train/gradients/CNN2/add_grad/Sum_1Sum'train/gradients/CNN2/Relu_grad/ReluGrad5train/gradients/CNN2/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
'train/gradients/CNN2/add_grad/Reshape_1Reshape#train/gradients/CNN2/add_grad/Sum_1%train/gradients/CNN2/add_grad/Shape_1"/device:GPU:0*
_output_shapes
: *
T0*
Tshape0
�
.train/gradients/CNN2/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1&^train/gradients/CNN2/add_grad/Reshape(^train/gradients/CNN2/add_grad/Reshape_1"/device:GPU:0
�
6train/gradients/CNN2/add_grad/tuple/control_dependencyIdentity%train/gradients/CNN2/add_grad/Reshape/^train/gradients/CNN2/add_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������   *
T0*8
_class.
,*loc:@train/gradients/CNN2/add_grad/Reshape
�
8train/gradients/CNN2/add_grad/tuple/control_dependency_1Identity'train/gradients/CNN2/add_grad/Reshape_1/^train/gradients/CNN2/add_grad/tuple/group_deps"/device:GPU:0*
T0*:
_class0
.,loc:@train/gradients/CNN2/add_grad/Reshape_1*
_output_shapes
: 
�
'train/gradients/CNN2/Conv2D_grad/ShapeNShapeN#CNN1/batch_normalization/cond/MergeCNN2/weights/Variable/read)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
::*
T0*
out_type0*
N
�
4train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'train/gradients/CNN2/Conv2D_grad/ShapeNCNN2/weights/Variable/read6train/gradients/CNN2/add_grad/tuple/control_dependency"/device:GPU:0*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
5train/gradients/CNN2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter#CNN1/batch_normalization/cond/Merge)train/gradients/CNN2/Conv2D_grad/ShapeN:16train/gradients/CNN2/add_grad/tuple/control_dependency"/device:GPU:0*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
1train/gradients/CNN2/Conv2D_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_15^train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput6^train/gradients/CNN2/Conv2D_grad/Conv2DBackpropFilter"/device:GPU:0
�
9train/gradients/CNN2/Conv2D_grad/tuple/control_dependencyIdentity4train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput2^train/gradients/CNN2/Conv2D_grad/tuple/group_deps"/device:GPU:0*
T0*G
_class=
;9loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������@@
�
;train/gradients/CNN2/Conv2D_grad/tuple/control_dependency_1Identity5train/gradients/CNN2/Conv2D_grad/Conv2DBackpropFilter2^train/gradients/CNN2/Conv2D_grad/tuple/group_deps"/device:GPU:0*&
_output_shapes
: *
T0*H
_class>
<:loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropFilter
�
Btrain/gradients/CNN1/batch_normalization/cond/Merge_grad/cond_gradSwitch9train/gradients/CNN2/Conv2D_grad/tuple/control_dependency%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*
T0*G
_class=
;9loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput*J
_output_shapes8
6:���������@@:���������@@
�
Itrain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/CNN1/batch_normalization/cond/Merge_grad/cond_grad"/device:GPU:0
�
Qtrain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentityBtrain/gradients/CNN1/batch_normalization/cond/Merge_grad/cond_gradJ^train/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������@@*
T0*G
_class=
;9loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput
�
Strain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/control_dependency_1IdentityDtrain/gradients/CNN1/batch_normalization/cond/Merge_grad/cond_grad:1J^train/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*G
_class=
;9loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������@@*
T0
�
train/gradients/zeros_like_17	ZerosLike0CNN1/batch_normalization/cond/FusedBatchNorm_1:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:@*
T0
�
train/gradients/zeros_like_18	ZerosLike0CNN1/batch_normalization/cond/FusedBatchNorm_1:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:@*
T0
�
train/gradients/zeros_like_19	ZerosLike0CNN1/batch_normalization/cond/FusedBatchNorm_1:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:@
�
train/gradients/zeros_like_20	ZerosLike0CNN1/batch_normalization/cond/FusedBatchNorm_1:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:@
�
Rtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Mtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose	Transpose5CNN1/batch_normalization/cond/FusedBatchNorm_1/SwitchRtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/perm"/device:GPU:0*
T0*/
_output_shapes
:���������@@*
Tperm0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*%
valueB"             *
dtype0
�
Otrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1	TransposeQtrain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/control_dependencyTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/perm"/device:GPU:0*/
_output_shapes
:���������@@*
Tperm0*
T0
�
Vtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradOtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1Mtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
data_formatNHWC*G
_output_shapes5
3:���������@@:@:@:@:@*
is_training( *
epsilon%o�:*
T0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Otrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2	TransposeVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/perm"/device:GPU:0*/
_output_shapes
:���������@@*
Tperm0*
T0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1W^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradP^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2"/device:GPU:0
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityOtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2U^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*b
_classX
VTloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2*/
_output_shapes
:���������@@
�
^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityXtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1U^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:@
�
^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityXtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2U^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:@
�
train/gradients/zeros_like_21	ZerosLike.CNN1/batch_normalization/cond/FusedBatchNorm:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:@*
T0
�
train/gradients/zeros_like_22	ZerosLike.CNN1/batch_normalization/cond/FusedBatchNorm:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:@*
T0
�
train/gradients/zeros_like_23	ZerosLike.CNN1/batch_normalization/cond/FusedBatchNorm:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:@*
T0
�
train/gradients/zeros_like_24	ZerosLike.CNN1/batch_normalization/cond/FusedBatchNorm:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:@*
T0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradStrain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/control_dependency_15CNN1/batch_normalization/cond/FusedBatchNorm/Switch:17CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1:1.CNN1/batch_normalization/cond/FusedBatchNorm:3.CNN1/batch_normalization/cond/FusedBatchNorm:4"/device:GPU:0*C
_output_shapes1
/:���������@@:@:@: : *
is_training(*
epsilon%o�:*
T0*
data_formatNCHW
�
Rtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1U^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad"/device:GPU:0
�
Ztrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradS^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������@@
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:@*
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
train/gradients/Switch_12Switch	CNN1/Relu%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*J
_output_shapes8
6:���������@@:���������@@
�
train/gradients/Shape_13Shapetrain/gradients/Switch_12:1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
train/gradients/zeros_12/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_12Filltrain/gradients/Shape_13train/gradients/zeros_12/Const"/device:GPU:0*
T0*/
_output_shapes
:���������@@
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMerge\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencytrain/gradients/zeros_12"/device:GPU:0*
N*1
_output_shapes
:���������@@: *
T0
�
train/gradients/Switch_13Switchbatch_normalization/gamma/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
:@:@
�
train/gradients/Shape_14Shapetrain/gradients/Switch_13:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_13/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_13Filltrain/gradients/Shape_14train/gradients/zeros_13/Const"/device:GPU:0*
T0*
_output_shapes
:@
�
Vtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMerge^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1train/gradients/zeros_13"/device:GPU:0*
T0*
N*
_output_shapes

:@: 
�
train/gradients/Switch_14Switchbatch_normalization/beta/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
:@:@*
T0
�
train/gradients/Shape_15Shapetrain/gradients/Switch_14:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_14/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_14Filltrain/gradients/Shape_15train/gradients/zeros_14/Const"/device:GPU:0*
T0*
_output_shapes
:@
�
Vtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMerge^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2train/gradients/zeros_14"/device:GPU:0*
N*
_output_shapes

:@: *
T0
�
train/gradients/Switch_15Switch	CNN1/Relu%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*J
_output_shapes8
6:���������@@:���������@@*
T0
�
train/gradients/Shape_16Shapetrain/gradients/Switch_15"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
train/gradients/zeros_15/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_15Filltrain/gradients/Shape_16train/gradients/zeros_15/Const"/device:GPU:0*
T0*/
_output_shapes
:���������@@
�
Rtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergeZtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencytrain/gradients/zeros_15"/device:GPU:0*
T0*
N*1
_output_shapes
:���������@@: 
�
train/gradients/Switch_16Switchbatch_normalization/gamma/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
:@:@*
T0
�
train/gradients/Shape_17Shapetrain/gradients/Switch_16"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
train/gradients/zeros_16/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_16Filltrain/gradients/Shape_17train/gradients/zeros_16/Const"/device:GPU:0*
_output_shapes
:@*
T0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMerge\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1train/gradients/zeros_16"/device:GPU:0*
T0*
N*
_output_shapes

:@: 
�
train/gradients/Switch_17Switchbatch_normalization/beta/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
:@:@
�
train/gradients/Shape_18Shapetrain/gradients/Switch_17"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_17/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
train/gradients/zeros_17Filltrain/gradients/Shape_18train/gradients/zeros_17/Const"/device:GPU:0*
T0*
_output_shapes
:@
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMerge\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2train/gradients/zeros_17"/device:GPU:0*
_output_shapes

:@: *
T0*
N
�
train/gradients/AddN_8AddNTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradRtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������@@
�
'train/gradients/CNN1/Relu_grad/ReluGradReluGradtrain/gradients/AddN_8	CNN1/Relu"/device:GPU:0*
T0*/
_output_shapes
:���������@@
�
train/gradients/AddN_9AddNVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad"/device:GPU:0*
_output_shapes
:@*
T0*i
_class_
][loc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N
�
train/gradients/AddN_10AddNVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad"/device:GPU:0*
_output_shapes
:@*
T0*i
_class_
][loc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N
�
#train/gradients/CNN1/add_grad/ShapeShapeCNN1/Conv2D)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
%train/gradients/CNN1/add_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:*
dtype0
�
3train/gradients/CNN1/add_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/CNN1/add_grad/Shape%train/gradients/CNN1/add_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
!train/gradients/CNN1/add_grad/SumSum'train/gradients/CNN1/Relu_grad/ReluGrad3train/gradients/CNN1/add_grad/BroadcastGradientArgs"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
%train/gradients/CNN1/add_grad/ReshapeReshape!train/gradients/CNN1/add_grad/Sum#train/gradients/CNN1/add_grad/Shape"/device:GPU:0*/
_output_shapes
:���������@@*
T0*
Tshape0
�
#train/gradients/CNN1/add_grad/Sum_1Sum'train/gradients/CNN1/Relu_grad/ReluGrad5train/gradients/CNN1/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
'train/gradients/CNN1/add_grad/Reshape_1Reshape#train/gradients/CNN1/add_grad/Sum_1%train/gradients/CNN1/add_grad/Shape_1"/device:GPU:0*
Tshape0*
_output_shapes
:*
T0
�
.train/gradients/CNN1/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1&^train/gradients/CNN1/add_grad/Reshape(^train/gradients/CNN1/add_grad/Reshape_1"/device:GPU:0
�
6train/gradients/CNN1/add_grad/tuple/control_dependencyIdentity%train/gradients/CNN1/add_grad/Reshape/^train/gradients/CNN1/add_grad/tuple/group_deps"/device:GPU:0*
T0*8
_class.
,*loc:@train/gradients/CNN1/add_grad/Reshape*/
_output_shapes
:���������@@
�
8train/gradients/CNN1/add_grad/tuple/control_dependency_1Identity'train/gradients/CNN1/add_grad/Reshape_1/^train/gradients/CNN1/add_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:*
T0*:
_class0
.,loc:@train/gradients/CNN1/add_grad/Reshape_1
�
'train/gradients/CNN1/Conv2D_grad/ShapeNShapeNinput/imagesCNN1/weights/Variable/read)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
N* 
_output_shapes
::*
T0
�
4train/gradients/CNN1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'train/gradients/CNN1/Conv2D_grad/ShapeNCNN1/weights/Variable/read6train/gradients/CNN1/add_grad/tuple/control_dependency"/device:GPU:0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0
�
5train/gradients/CNN1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput/images)train/gradients/CNN1/Conv2D_grad/ShapeN:16train/gradients/CNN1/add_grad/tuple/control_dependency"/device:GPU:0*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
1train/gradients/CNN1/Conv2D_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_15^train/gradients/CNN1/Conv2D_grad/Conv2DBackpropInput6^train/gradients/CNN1/Conv2D_grad/Conv2DBackpropFilter"/device:GPU:0
�
9train/gradients/CNN1/Conv2D_grad/tuple/control_dependencyIdentity4train/gradients/CNN1/Conv2D_grad/Conv2DBackpropInput2^train/gradients/CNN1/Conv2D_grad/tuple/group_deps"/device:GPU:0*
T0*G
_class=
;9loc:@train/gradients/CNN1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������@@
�
;train/gradients/CNN1/Conv2D_grad/tuple/control_dependency_1Identity5train/gradients/CNN1/Conv2D_grad/Conv2DBackpropFilter2^train/gradients/CNN1/Conv2D_grad/tuple/group_deps"/device:GPU:0*
T0*H
_class>
<:loc:@train/gradients/CNN1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
�
train/beta1_power/initial_valueConst"/device:GPU:0*
valueB
 *fff?*'
_class
loc:@CNN1/biases/Variable*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2"/device:GPU:0*
shared_name *'
_class
loc:@CNN1/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
train/beta1_power/readIdentitytrain/beta1_power"/device:GPU:0*
T0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
: 
�
train/beta2_power/initial_valueConst"/device:GPU:0*
_output_shapes
: *
valueB
 *w�?*'
_class
loc:@CNN1/biases/Variable*
dtype0
�
train/beta2_power
VariableV2"/device:GPU:0*
_output_shapes
: *
shared_name *'
_class
loc:@CNN1/biases/Variable*
	container *
shape: *
dtype0
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value"/device:GPU:0*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(
�
train/beta2_power/readIdentitytrain/beta2_power"/device:GPU:0*
_output_shapes
: *
T0*'
_class
loc:@CNN1/biases/Variable
�
,CNN1/weights/Variable/Adam/Initializer/zerosConst"/device:GPU:0*(
_class
loc:@CNN1/weights/Variable*%
valueB*    *
dtype0*&
_output_shapes
:
�
CNN1/weights/Variable/Adam
VariableV2"/device:GPU:0*(
_class
loc:@CNN1/weights/Variable*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name 
�
!CNN1/weights/Variable/Adam/AssignAssignCNN1/weights/Variable/Adam,CNN1/weights/Variable/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN1/weights/Variable*
validate_shape(*&
_output_shapes
:
�
CNN1/weights/Variable/Adam/readIdentityCNN1/weights/Variable/Adam"/device:GPU:0*(
_class
loc:@CNN1/weights/Variable*&
_output_shapes
:*
T0
�
.CNN1/weights/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*(
_class
loc:@CNN1/weights/Variable*%
valueB*    *
dtype0*&
_output_shapes
:
�
CNN1/weights/Variable/Adam_1
VariableV2"/device:GPU:0*(
_class
loc:@CNN1/weights/Variable*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name 
�
#CNN1/weights/Variable/Adam_1/AssignAssignCNN1/weights/Variable/Adam_1.CNN1/weights/Variable/Adam_1/Initializer/zeros"/device:GPU:0*&
_output_shapes
:*
use_locking(*
T0*(
_class
loc:@CNN1/weights/Variable*
validate_shape(
�
!CNN1/weights/Variable/Adam_1/readIdentityCNN1/weights/Variable/Adam_1"/device:GPU:0*
T0*(
_class
loc:@CNN1/weights/Variable*&
_output_shapes
:
�
+CNN1/biases/Variable/Adam/Initializer/zerosConst"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
valueB*    *
dtype0*
_output_shapes
:
�
CNN1/biases/Variable/Adam
VariableV2"/device:GPU:0*
shared_name *'
_class
loc:@CNN1/biases/Variable*
	container *
shape:*
dtype0*
_output_shapes
:
�
 CNN1/biases/Variable/Adam/AssignAssignCNN1/biases/Variable/Adam+CNN1/biases/Variable/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
:
�
CNN1/biases/Variable/Adam/readIdentityCNN1/biases/Variable/Adam"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
:*
T0
�
-CNN1/biases/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
valueB*    *
dtype0*
_output_shapes
:
�
CNN1/biases/Variable/Adam_1
VariableV2"/device:GPU:0*
shape:*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@CNN1/biases/Variable*
	container 
�
"CNN1/biases/Variable/Adam_1/AssignAssignCNN1/biases/Variable/Adam_1-CNN1/biases/Variable/Adam_1/Initializer/zeros"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
 CNN1/biases/Variable/Adam_1/readIdentityCNN1/biases/Variable/Adam_1"/device:GPU:0*
_output_shapes
:*
T0*'
_class
loc:@CNN1/biases/Variable
�
0batch_normalization/gamma/Adam/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:@*,
_class"
 loc:@batch_normalization/gamma*
valueB@*    *
dtype0
�
batch_normalization/gamma/Adam
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:@*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:@
�
%batch_normalization/gamma/Adam/AssignAssignbatch_normalization/gamma/Adam0batch_normalization/gamma/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:@
�
#batch_normalization/gamma/Adam/readIdentitybatch_normalization/gamma/Adam"/device:GPU:0*
_output_shapes
:@*
T0*,
_class"
 loc:@batch_normalization/gamma
�
2batch_normalization/gamma/Adam_1/Initializer/zerosConst"/device:GPU:0*,
_class"
 loc:@batch_normalization/gamma*
valueB@*    *
dtype0*
_output_shapes
:@
�
 batch_normalization/gamma/Adam_1
VariableV2"/device:GPU:0*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
'batch_normalization/gamma/Adam_1/AssignAssign batch_normalization/gamma/Adam_12batch_normalization/gamma/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(
�
%batch_normalization/gamma/Adam_1/readIdentity batch_normalization/gamma/Adam_1"/device:GPU:0*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
�
/batch_normalization/beta/Adam/Initializer/zerosConst"/device:GPU:0*+
_class!
loc:@batch_normalization/beta*
valueB@*    *
dtype0*
_output_shapes
:@
�
batch_normalization/beta/Adam
VariableV2"/device:GPU:0*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container 
�
$batch_normalization/beta/Adam/AssignAssignbatch_normalization/beta/Adam/batch_normalization/beta/Adam/Initializer/zeros"/device:GPU:0*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(
�
"batch_normalization/beta/Adam/readIdentitybatch_normalization/beta/Adam"/device:GPU:0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
:@*
T0
�
1batch_normalization/beta/Adam_1/Initializer/zerosConst"/device:GPU:0*+
_class!
loc:@batch_normalization/beta*
valueB@*    *
dtype0*
_output_shapes
:@
�
batch_normalization/beta/Adam_1
VariableV2"/device:GPU:0*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container 
�
&batch_normalization/beta/Adam_1/AssignAssignbatch_normalization/beta/Adam_11batch_normalization/beta/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(
�
$batch_normalization/beta/Adam_1/readIdentitybatch_normalization/beta/Adam_1"/device:GPU:0*
_output_shapes
:@*
T0*+
_class!
loc:@batch_normalization/beta
�
,CNN2/weights/Variable/Adam/Initializer/zerosConst"/device:GPU:0*(
_class
loc:@CNN2/weights/Variable*%
valueB *    *
dtype0*&
_output_shapes
: 
�
CNN2/weights/Variable/Adam
VariableV2"/device:GPU:0*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name *(
_class
loc:@CNN2/weights/Variable
�
!CNN2/weights/Variable/Adam/AssignAssignCNN2/weights/Variable/Adam,CNN2/weights/Variable/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN2/weights/Variable*
validate_shape(*&
_output_shapes
: 
�
CNN2/weights/Variable/Adam/readIdentityCNN2/weights/Variable/Adam"/device:GPU:0*
T0*(
_class
loc:@CNN2/weights/Variable*&
_output_shapes
: 
�
.CNN2/weights/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*(
_class
loc:@CNN2/weights/Variable*%
valueB *    *
dtype0*&
_output_shapes
: 
�
CNN2/weights/Variable/Adam_1
VariableV2"/device:GPU:0*
dtype0*&
_output_shapes
: *
shared_name *(
_class
loc:@CNN2/weights/Variable*
	container *
shape: 
�
#CNN2/weights/Variable/Adam_1/AssignAssignCNN2/weights/Variable/Adam_1.CNN2/weights/Variable/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN2/weights/Variable*
validate_shape(*&
_output_shapes
: 
�
!CNN2/weights/Variable/Adam_1/readIdentityCNN2/weights/Variable/Adam_1"/device:GPU:0*
T0*(
_class
loc:@CNN2/weights/Variable*&
_output_shapes
: 
�
+CNN2/biases/Variable/Adam/Initializer/zerosConst"/device:GPU:0*'
_class
loc:@CNN2/biases/Variable*
valueB *    *
dtype0*
_output_shapes
: 
�
CNN2/biases/Variable/Adam
VariableV2"/device:GPU:0*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@CNN2/biases/Variable
�
 CNN2/biases/Variable/Adam/AssignAssignCNN2/biases/Variable/Adam+CNN2/biases/Variable/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN2/biases/Variable*
validate_shape(*
_output_shapes
: 
�
CNN2/biases/Variable/Adam/readIdentityCNN2/biases/Variable/Adam"/device:GPU:0*'
_class
loc:@CNN2/biases/Variable*
_output_shapes
: *
T0
�
-CNN2/biases/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*'
_class
loc:@CNN2/biases/Variable*
valueB *    *
dtype0*
_output_shapes
: 
�
CNN2/biases/Variable/Adam_1
VariableV2"/device:GPU:0*
shared_name *'
_class
loc:@CNN2/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
�
"CNN2/biases/Variable/Adam_1/AssignAssignCNN2/biases/Variable/Adam_1-CNN2/biases/Variable/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN2/biases/Variable*
validate_shape(*
_output_shapes
: 
�
 CNN2/biases/Variable/Adam_1/readIdentityCNN2/biases/Variable/Adam_1"/device:GPU:0*
T0*'
_class
loc:@CNN2/biases/Variable*
_output_shapes
: 
�
2batch_normalization_1/gamma/Adam/Initializer/zerosConst"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma*
valueB *    *
dtype0*
_output_shapes
: 
�
 batch_normalization_1/gamma/Adam
VariableV2"/device:GPU:0*
shape: *
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container 
�
'batch_normalization_1/gamma/Adam/AssignAssign batch_normalization_1/gamma/Adam2batch_normalization_1/gamma/Adam/Initializer/zeros"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
%batch_normalization_1/gamma/Adam/readIdentity batch_normalization_1/gamma/Adam"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: 
�
4batch_normalization_1/gamma/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_1/gamma*
valueB *    *
dtype0
�
"batch_normalization_1/gamma/Adam_1
VariableV2"/device:GPU:0*
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape: *
dtype0*
_output_shapes
: 
�
)batch_normalization_1/gamma/Adam_1/AssignAssign"batch_normalization_1/gamma/Adam_14batch_normalization_1/gamma/Adam_1/Initializer/zeros"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
'batch_normalization_1/gamma/Adam_1/readIdentity"batch_normalization_1/gamma/Adam_1"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: *
T0
�
1batch_normalization_1/beta/Adam/Initializer/zerosConst"/device:GPU:0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_1/beta*
valueB *    *
dtype0
�
batch_normalization_1/beta/Adam
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
: *
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape: 
�
&batch_normalization_1/beta/Adam/AssignAssignbatch_normalization_1/beta/Adam1batch_normalization_1/beta/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes
: 
�
$batch_normalization_1/beta/Adam/readIdentitybatch_normalization_1/beta/Adam"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: 
�
3batch_normalization_1/beta/Adam_1/Initializer/zerosConst"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta*
valueB *    *
dtype0*
_output_shapes
: 
�
!batch_normalization_1/beta/Adam_1
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
: *
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape: 
�
(batch_normalization_1/beta/Adam_1/AssignAssign!batch_normalization_1/beta/Adam_13batch_normalization_1/beta/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes
: *
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(
�
&batch_normalization_1/beta/Adam_1/readIdentity!batch_normalization_1/beta/Adam_1"/device:GPU:0*
_output_shapes
: *
T0*-
_class#
!loc:@batch_normalization_1/beta
�
,CNN3/weights/Variable/Adam/Initializer/zerosConst"/device:GPU:0*(
_class
loc:@CNN3/weights/Variable*%
valueB @*    *
dtype0*&
_output_shapes
: @
�
CNN3/weights/Variable/Adam
VariableV2"/device:GPU:0*
	container *
shape: @*
dtype0*&
_output_shapes
: @*
shared_name *(
_class
loc:@CNN3/weights/Variable
�
!CNN3/weights/Variable/Adam/AssignAssignCNN3/weights/Variable/Adam,CNN3/weights/Variable/Adam/Initializer/zeros"/device:GPU:0*(
_class
loc:@CNN3/weights/Variable*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0
�
CNN3/weights/Variable/Adam/readIdentityCNN3/weights/Variable/Adam"/device:GPU:0*&
_output_shapes
: @*
T0*(
_class
loc:@CNN3/weights/Variable
�
.CNN3/weights/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*(
_class
loc:@CNN3/weights/Variable*%
valueB @*    *
dtype0*&
_output_shapes
: @
�
CNN3/weights/Variable/Adam_1
VariableV2"/device:GPU:0*&
_output_shapes
: @*
shared_name *(
_class
loc:@CNN3/weights/Variable*
	container *
shape: @*
dtype0
�
#CNN3/weights/Variable/Adam_1/AssignAssignCNN3/weights/Variable/Adam_1.CNN3/weights/Variable/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN3/weights/Variable*
validate_shape(*&
_output_shapes
: @
�
!CNN3/weights/Variable/Adam_1/readIdentityCNN3/weights/Variable/Adam_1"/device:GPU:0*
T0*(
_class
loc:@CNN3/weights/Variable*&
_output_shapes
: @
�
+CNN3/biases/Variable/Adam/Initializer/zerosConst"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
valueB@*    *
dtype0*
_output_shapes
:@
�
CNN3/biases/Variable/Adam
VariableV2"/device:GPU:0*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@CNN3/biases/Variable
�
 CNN3/biases/Variable/Adam/AssignAssignCNN3/biases/Variable/Adam+CNN3/biases/Variable/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN3/biases/Variable*
validate_shape(*
_output_shapes
:@
�
CNN3/biases/Variable/Adam/readIdentityCNN3/biases/Variable/Adam"/device:GPU:0*
T0*'
_class
loc:@CNN3/biases/Variable*
_output_shapes
:@
�
-CNN3/biases/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:@*'
_class
loc:@CNN3/biases/Variable*
valueB@*    *
dtype0
�
CNN3/biases/Variable/Adam_1
VariableV2"/device:GPU:0*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@CNN3/biases/Variable*
	container 
�
"CNN3/biases/Variable/Adam_1/AssignAssignCNN3/biases/Variable/Adam_1-CNN3/biases/Variable/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@CNN3/biases/Variable*
validate_shape(
�
 CNN3/biases/Variable/Adam_1/readIdentityCNN3/biases/Variable/Adam_1"/device:GPU:0*
_output_shapes
:@*
T0*'
_class
loc:@CNN3/biases/Variable
�
2batch_normalization_2/gamma/Adam/Initializer/zerosConst"/device:GPU:0*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
 batch_normalization_2/gamma/Adam
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:
�
'batch_normalization_2/gamma/Adam/AssignAssign batch_normalization_2/gamma/Adam2batch_normalization_2/gamma/Adam/Initializer/zeros"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(
�
%batch_normalization_2/gamma/Adam/readIdentity batch_normalization_2/gamma/Adam"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
�
4batch_normalization_2/gamma/Adam_1/Initializer/zerosConst"/device:GPU:0*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*    *
dtype0*
_output_shapes
:
�
"batch_normalization_2/gamma/Adam_1
VariableV2"/device:GPU:0*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma
�
)batch_normalization_2/gamma/Adam_1/AssignAssign"batch_normalization_2/gamma/Adam_14batch_normalization_2/gamma/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(
�
'batch_normalization_2/gamma/Adam_1/readIdentity"batch_normalization_2/gamma/Adam_1"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
�
1batch_normalization_2/beta/Adam/Initializer/zerosConst"/device:GPU:0*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
dtype0*
_output_shapes
:
�
batch_normalization_2/beta/Adam
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:
�
&batch_normalization_2/beta/Adam/AssignAssignbatch_normalization_2/beta/Adam1batch_normalization_2/beta/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
:
�
$batch_normalization_2/beta/Adam/readIdentitybatch_normalization_2/beta/Adam"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:
�
3batch_normalization_2/beta/Adam_1/Initializer/zerosConst"/device:GPU:0*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
dtype0*
_output_shapes
:
�
!batch_normalization_2/beta/Adam_1
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:
�
(batch_normalization_2/beta/Adam_1/AssignAssign!batch_normalization_2/beta/Adam_13batch_normalization_2/beta/Adam_1/Initializer/zeros"/device:GPU:0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
&batch_normalization_2/beta/Adam_1/readIdentity!batch_normalization_2/beta/Adam_1"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:
�
#FC1/Variable/Adam/Initializer/zerosConst"/device:GPU:0*!
_output_shapes
:���*
_class
loc:@FC1/Variable* 
valueB���*    *
dtype0
�
FC1/Variable/Adam
VariableV2"/device:GPU:0*
_class
loc:@FC1/Variable*
	container *
shape:���*
dtype0*!
_output_shapes
:���*
shared_name 
�
FC1/Variable/Adam/AssignAssignFC1/Variable/Adam#FC1/Variable/Adam/Initializer/zeros"/device:GPU:0*
_class
loc:@FC1/Variable*
validate_shape(*!
_output_shapes
:���*
use_locking(*
T0
�
FC1/Variable/Adam/readIdentityFC1/Variable/Adam"/device:GPU:0*
_class
loc:@FC1/Variable*!
_output_shapes
:���*
T0
�
%FC1/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*
_class
loc:@FC1/Variable* 
valueB���*    *
dtype0*!
_output_shapes
:���
�
FC1/Variable/Adam_1
VariableV2"/device:GPU:0*
shared_name *
_class
loc:@FC1/Variable*
	container *
shape:���*
dtype0*!
_output_shapes
:���
�
FC1/Variable/Adam_1/AssignAssignFC1/Variable/Adam_1%FC1/Variable/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*
_class
loc:@FC1/Variable*
validate_shape(*!
_output_shapes
:���
�
FC1/Variable/Adam_1/readIdentityFC1/Variable/Adam_1"/device:GPU:0*!
_output_shapes
:���*
T0*
_class
loc:@FC1/Variable
�
%FC1/Variable_1/Adam/Initializer/zerosConst"/device:GPU:0*!
_class
loc:@FC1/Variable_1*
valueB�*    *
dtype0*
_output_shapes	
:�
�
FC1/Variable_1/Adam
VariableV2"/device:GPU:0*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *!
_class
loc:@FC1/Variable_1
�
FC1/Variable_1/Adam/AssignAssignFC1/Variable_1/Adam%FC1/Variable_1/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*!
_class
loc:@FC1/Variable_1*
validate_shape(*
_output_shapes	
:�
�
FC1/Variable_1/Adam/readIdentityFC1/Variable_1/Adam"/device:GPU:0*
T0*!
_class
loc:@FC1/Variable_1*
_output_shapes	
:�
�
'FC1/Variable_1/Adam_1/Initializer/zerosConst"/device:GPU:0*!
_class
loc:@FC1/Variable_1*
valueB�*    *
dtype0*
_output_shapes	
:�
�
FC1/Variable_1/Adam_1
VariableV2"/device:GPU:0*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *!
_class
loc:@FC1/Variable_1*
	container 
�
FC1/Variable_1/Adam_1/AssignAssignFC1/Variable_1/Adam_1'FC1/Variable_1/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*!
_class
loc:@FC1/Variable_1*
validate_shape(*
_output_shapes	
:�
�
FC1/Variable_1/Adam_1/readIdentityFC1/Variable_1/Adam_1"/device:GPU:0*
_output_shapes	
:�*
T0*!
_class
loc:@FC1/Variable_1
�
2batch_normalization_3/gamma/Adam/Initializer/zerosConst"/device:GPU:0*.
_class$
" loc:@batch_normalization_3/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
 batch_normalization_3/gamma/Adam
VariableV2"/device:GPU:0*.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
'batch_normalization_3/gamma/Adam/AssignAssign batch_normalization_3/gamma/Adam2batch_normalization_3/gamma/Adam/Initializer/zeros"/device:GPU:0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
%batch_normalization_3/gamma/Adam/readIdentity batch_normalization_3/gamma/Adam"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:�
�
4batch_normalization_3/gamma/Adam_1/Initializer/zerosConst"/device:GPU:0*.
_class$
" loc:@batch_normalization_3/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
"batch_normalization_3/gamma/Adam_1
VariableV2"/device:GPU:0*
_output_shapes	
:�*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container *
shape:�*
dtype0
�
)batch_normalization_3/gamma/Adam_1/AssignAssign"batch_normalization_3/gamma/Adam_14batch_normalization_3/gamma/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:�
�
'batch_normalization_3/gamma/Adam_1/readIdentity"batch_normalization_3/gamma/Adam_1"/device:GPU:0*
_output_shapes	
:�*
T0*.
_class$
" loc:@batch_normalization_3/gamma
�
1batch_normalization_3/beta/Adam/Initializer/zerosConst"/device:GPU:0*-
_class#
!loc:@batch_normalization_3/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
batch_normalization_3/beta/Adam
VariableV2"/device:GPU:0*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
&batch_normalization_3/beta/Adam/AssignAssignbatch_normalization_3/beta/Adam1batch_normalization_3/beta/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:�
�
$batch_normalization_3/beta/Adam/readIdentitybatch_normalization_3/beta/Adam"/device:GPU:0*
_output_shapes	
:�*
T0*-
_class#
!loc:@batch_normalization_3/beta
�
3batch_normalization_3/beta/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes	
:�*-
_class#
!loc:@batch_normalization_3/beta*
valueB�*    *
dtype0
�
!batch_normalization_3/beta/Adam_1
VariableV2"/device:GPU:0*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
(batch_normalization_3/beta/Adam_1/AssignAssign!batch_normalization_3/beta/Adam_13batch_normalization_3/beta/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:�
�
&batch_normalization_3/beta/Adam_1/readIdentity!batch_normalization_3/beta/Adam_1"/device:GPU:0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:�*
T0
�
'Readout/Variable/Adam/Initializer/zerosConst"/device:GPU:0*#
_class
loc:@Readout/Variable*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
Readout/Variable/Adam
VariableV2"/device:GPU:0*
shared_name *#
_class
loc:@Readout/Variable*
	container *
shape:	�*
dtype0*
_output_shapes
:	�
�
Readout/Variable/Adam/AssignAssignReadout/Variable/Adam'Readout/Variable/Adam/Initializer/zeros"/device:GPU:0*
_output_shapes
:	�*
use_locking(*
T0*#
_class
loc:@Readout/Variable*
validate_shape(
�
Readout/Variable/Adam/readIdentityReadout/Variable/Adam"/device:GPU:0*
_output_shapes
:	�*
T0*#
_class
loc:@Readout/Variable
�
)Readout/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*#
_class
loc:@Readout/Variable*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
Readout/Variable/Adam_1
VariableV2"/device:GPU:0*
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *#
_class
loc:@Readout/Variable*
	container 
�
Readout/Variable/Adam_1/AssignAssignReadout/Variable/Adam_1)Readout/Variable/Adam_1/Initializer/zeros"/device:GPU:0*#
_class
loc:@Readout/Variable*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0
�
Readout/Variable/Adam_1/readIdentityReadout/Variable/Adam_1"/device:GPU:0*
T0*#
_class
loc:@Readout/Variable*
_output_shapes
:	�
�
)Readout/Variable_1/Adam/Initializer/zerosConst"/device:GPU:0*%
_class
loc:@Readout/Variable_1*
valueB*    *
dtype0*
_output_shapes
:
�
Readout/Variable_1/Adam
VariableV2"/device:GPU:0*
_output_shapes
:*
shared_name *%
_class
loc:@Readout/Variable_1*
	container *
shape:*
dtype0
�
Readout/Variable_1/Adam/AssignAssignReadout/Variable_1/Adam)Readout/Variable_1/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*%
_class
loc:@Readout/Variable_1*
validate_shape(*
_output_shapes
:
�
Readout/Variable_1/Adam/readIdentityReadout/Variable_1/Adam"/device:GPU:0*
_output_shapes
:*
T0*%
_class
loc:@Readout/Variable_1
�
+Readout/Variable_1/Adam_1/Initializer/zerosConst"/device:GPU:0*%
_class
loc:@Readout/Variable_1*
valueB*    *
dtype0*
_output_shapes
:
�
Readout/Variable_1/Adam_1
VariableV2"/device:GPU:0*
shared_name *%
_class
loc:@Readout/Variable_1*
	container *
shape:*
dtype0*
_output_shapes
:
�
 Readout/Variable_1/Adam_1/AssignAssignReadout/Variable_1/Adam_1+Readout/Variable_1/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*%
_class
loc:@Readout/Variable_1*
validate_shape(*
_output_shapes
:
�
Readout/Variable_1/Adam_1/readIdentityReadout/Variable_1/Adam_1"/device:GPU:0*
T0*%
_class
loc:@Readout/Variable_1*
_output_shapes
:
�
train/Adam/learning_rateConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *��8*
dtype0*
_output_shapes
: 
�
train/Adam/beta1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
train/Adam/beta2Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *w�?*
dtype0
�
train/Adam/epsilonConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *w�+2*
dtype0
�
1train/Adam/update_CNN1/weights/Variable/ApplyAdam	ApplyAdamCNN1/weights/VariableCNN1/weights/Variable/AdamCNN1/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/CNN1/Conv2D_grad/tuple/control_dependency_1"/device:GPU:0*
use_locking( *
T0*(
_class
loc:@CNN1/weights/Variable*
use_nesterov( *&
_output_shapes
:
�
0train/Adam/update_CNN1/biases/Variable/ApplyAdam	ApplyAdamCNN1/biases/VariableCNN1/biases/Variable/AdamCNN1/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon8train/gradients/CNN1/add_grad/tuple/control_dependency_1"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
5train/Adam/update_batch_normalization/gamma/ApplyAdam	ApplyAdambatch_normalization/gammabatch_normalization/gamma/Adam batch_normalization/gamma/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_9"/device:GPU:0*
use_locking( *
T0*,
_class"
 loc:@batch_normalization/gamma*
use_nesterov( *
_output_shapes
:@
�
4train/Adam/update_batch_normalization/beta/ApplyAdam	ApplyAdambatch_normalization/betabatch_normalization/beta/Adambatch_normalization/beta/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_10"/device:GPU:0*
use_locking( *
T0*+
_class!
loc:@batch_normalization/beta*
use_nesterov( *
_output_shapes
:@
�
1train/Adam/update_CNN2/weights/Variable/ApplyAdam	ApplyAdamCNN2/weights/VariableCNN2/weights/Variable/AdamCNN2/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/CNN2/Conv2D_grad/tuple/control_dependency_1"/device:GPU:0*&
_output_shapes
: *
use_locking( *
T0*(
_class
loc:@CNN2/weights/Variable*
use_nesterov( 
�
0train/Adam/update_CNN2/biases/Variable/ApplyAdam	ApplyAdamCNN2/biases/VariableCNN2/biases/Variable/AdamCNN2/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon8train/gradients/CNN2/add_grad/tuple/control_dependency_1"/device:GPU:0*
_output_shapes
: *
use_locking( *
T0*'
_class
loc:@CNN2/biases/Variable*
use_nesterov( 
�
7train/Adam/update_batch_normalization_1/gamma/ApplyAdam	ApplyAdambatch_normalization_1/gamma batch_normalization_1/gamma/Adam"batch_normalization_1/gamma/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_6"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma*
use_nesterov( *
_output_shapes
: *
use_locking( *
T0
�
6train/Adam/update_batch_normalization_1/beta/ApplyAdam	ApplyAdambatch_normalization_1/betabatch_normalization_1/beta/Adam!batch_normalization_1/beta/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_7"/device:GPU:0*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_1/beta*
use_nesterov( *
_output_shapes
: 
�
1train/Adam/update_CNN3/weights/Variable/ApplyAdam	ApplyAdamCNN3/weights/VariableCNN3/weights/Variable/AdamCNN3/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/CNN3/Conv2D_grad/tuple/control_dependency_1"/device:GPU:0*(
_class
loc:@CNN3/weights/Variable*
use_nesterov( *&
_output_shapes
: @*
use_locking( *
T0
�
0train/Adam/update_CNN3/biases/Variable/ApplyAdam	ApplyAdamCNN3/biases/VariableCNN3/biases/Variable/AdamCNN3/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon8train/gradients/CNN3/add_grad/tuple/control_dependency_1"/device:GPU:0*
_output_shapes
:@*
use_locking( *
T0*'
_class
loc:@CNN3/biases/Variable*
use_nesterov( 
�
7train/Adam/update_batch_normalization_2/gamma/ApplyAdam	ApplyAdambatch_normalization_2/gamma batch_normalization_2/gamma/Adam"batch_normalization_2/gamma/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_3"/device:GPU:0*
use_locking( *
T0*.
_class$
" loc:@batch_normalization_2/gamma*
use_nesterov( *
_output_shapes
:
�
6train/Adam/update_batch_normalization_2/beta/ApplyAdam	ApplyAdambatch_normalization_2/betabatch_normalization_2/beta/Adam!batch_normalization_2/beta/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_4"/device:GPU:0*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_2/beta*
use_nesterov( *
_output_shapes
:
�
(train/Adam/update_FC1/Variable/ApplyAdam	ApplyAdamFC1/VariableFC1/Variable/AdamFC1/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/FC1/MatMul_grad/tuple/control_dependency_1"/device:GPU:0*!
_output_shapes
:���*
use_locking( *
T0*
_class
loc:@FC1/Variable*
use_nesterov( 
�
*train/Adam/update_FC1/Variable_1/ApplyAdam	ApplyAdamFC1/Variable_1FC1/Variable_1/AdamFC1/Variable_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon7train/gradients/FC1/add_grad/tuple/control_dependency_1"/device:GPU:0*!
_class
loc:@FC1/Variable_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
7train/Adam/update_batch_normalization_3/gamma/ApplyAdam	ApplyAdambatch_normalization_3/gamma batch_normalization_3/gamma/Adam"batch_normalization_3/gamma/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1"/device:GPU:0*.
_class$
" loc:@batch_normalization_3/gamma*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
6train/Adam/update_batch_normalization_3/beta/ApplyAdam	ApplyAdambatch_normalization_3/betabatch_normalization_3/beta/Adam!batch_normalization_3/beta/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonStrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependency"/device:GPU:0*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_3/beta*
use_nesterov( *
_output_shapes	
:�
�
,train/Adam/update_Readout/Variable/ApplyAdam	ApplyAdamReadout/VariableReadout/Variable/AdamReadout/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/Readout/MatMul_grad/tuple/control_dependency_1"/device:GPU:0*
_output_shapes
:	�*
use_locking( *
T0*#
_class
loc:@Readout/Variable*
use_nesterov( 
�
.train/Adam/update_Readout/Variable_1/ApplyAdam	ApplyAdamReadout/Variable_1Readout/Variable_1/AdamReadout/Variable_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonAtrain/gradients/Readout/predicted_grad/tuple/control_dependency_1"/device:GPU:0*
use_locking( *
T0*%
_class
loc:@Readout/Variable_1*
use_nesterov( *
_output_shapes
:
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta12^train/Adam/update_CNN1/weights/Variable/ApplyAdam1^train/Adam/update_CNN1/biases/Variable/ApplyAdam6^train/Adam/update_batch_normalization/gamma/ApplyAdam5^train/Adam/update_batch_normalization/beta/ApplyAdam2^train/Adam/update_CNN2/weights/Variable/ApplyAdam1^train/Adam/update_CNN2/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_1/gamma/ApplyAdam7^train/Adam/update_batch_normalization_1/beta/ApplyAdam2^train/Adam/update_CNN3/weights/Variable/ApplyAdam1^train/Adam/update_CNN3/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_2/gamma/ApplyAdam7^train/Adam/update_batch_normalization_2/beta/ApplyAdam)^train/Adam/update_FC1/Variable/ApplyAdam+^train/Adam/update_FC1/Variable_1/ApplyAdam8^train/Adam/update_batch_normalization_3/gamma/ApplyAdam7^train/Adam/update_batch_normalization_3/beta/ApplyAdam-^train/Adam/update_Readout/Variable/ApplyAdam/^train/Adam/update_Readout/Variable_1/ApplyAdam"/device:GPU:0*
T0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul"/device:GPU:0*
use_locking( *
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta22^train/Adam/update_CNN1/weights/Variable/ApplyAdam1^train/Adam/update_CNN1/biases/Variable/ApplyAdam6^train/Adam/update_batch_normalization/gamma/ApplyAdam5^train/Adam/update_batch_normalization/beta/ApplyAdam2^train/Adam/update_CNN2/weights/Variable/ApplyAdam1^train/Adam/update_CNN2/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_1/gamma/ApplyAdam7^train/Adam/update_batch_normalization_1/beta/ApplyAdam2^train/Adam/update_CNN3/weights/Variable/ApplyAdam1^train/Adam/update_CNN3/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_2/gamma/ApplyAdam7^train/Adam/update_batch_normalization_2/beta/ApplyAdam)^train/Adam/update_FC1/Variable/ApplyAdam+^train/Adam/update_FC1/Variable_1/ApplyAdam8^train/Adam/update_batch_normalization_3/gamma/ApplyAdam7^train/Adam/update_batch_normalization_3/beta/ApplyAdam-^train/Adam/update_Readout/Variable/ApplyAdam/^train/Adam/update_Readout/Variable_1/ApplyAdam"/device:GPU:0*
_output_shapes
: *
T0*'
_class
loc:@CNN1/biases/Variable
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1"/device:GPU:0*
_output_shapes
: *
use_locking( *
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(
�


train/AdamNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_12^train/Adam/update_CNN1/weights/Variable/ApplyAdam1^train/Adam/update_CNN1/biases/Variable/ApplyAdam6^train/Adam/update_batch_normalization/gamma/ApplyAdam5^train/Adam/update_batch_normalization/beta/ApplyAdam2^train/Adam/update_CNN2/weights/Variable/ApplyAdam1^train/Adam/update_CNN2/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_1/gamma/ApplyAdam7^train/Adam/update_batch_normalization_1/beta/ApplyAdam2^train/Adam/update_CNN3/weights/Variable/ApplyAdam1^train/Adam/update_CNN3/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_2/gamma/ApplyAdam7^train/Adam/update_batch_normalization_2/beta/ApplyAdam)^train/Adam/update_FC1/Variable/ApplyAdam+^train/Adam/update_FC1/Variable_1/ApplyAdam8^train/Adam/update_batch_normalization_3/gamma/ApplyAdam7^train/Adam/update_batch_normalization_3/beta/ApplyAdam-^train/Adam/update_Readout/Variable/ApplyAdam/^train/Adam/update_Readout/Variable_1/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1"/device:GPU:0
j
accuracy/ArgMax/dimensionConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
accuracy/ArgMaxArgMaxReadout/predictedaccuracy/ArgMax/dimension"/device:GPU:0*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
l
accuracy/ArgMax_1/dimensionConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
accuracy/ArgMax_1ArgMaxinput/correct_labelsaccuracy/ArgMax_1/dimension"/device:GPU:0*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0

accuracy/correct_predEqualaccuracy/ArgMaxaccuracy/ArgMax_1"/device:GPU:0*#
_output_shapes
:���������*
T0	
x
accuracy/CastCastaccuracy/correct_pred"/device:GPU:0*#
_output_shapes
:���������*

DstT0*

SrcT0

g
accuracy/ConstConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
accuracy/accuracyMeanaccuracy/Castaccuracy/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
4accuracy_streaming_mean/mean/total/Initializer/zerosConst*
_output_shapes
: *5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
valueB
 *    *
dtype0
�
"accuracy_streaming_mean/mean/total
VariableV2"/device:GPU:0*
shared_name *5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
	container *
shape: *
dtype0*
_output_shapes
: 
�
)accuracy_streaming_mean/mean/total/AssignAssign"accuracy_streaming_mean/mean/total4accuracy_streaming_mean/mean/total/Initializer/zeros"/device:GPU:0*
_output_shapes
: *
use_locking(*
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
validate_shape(
�
'accuracy_streaming_mean/mean/total/readIdentity"accuracy_streaming_mean/mean/total"/device:GPU:0*5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
_output_shapes
: *
T0
�
4accuracy_streaming_mean/mean/count/Initializer/zerosConst*5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"accuracy_streaming_mean/mean/count
VariableV2"/device:GPU:0*
shared_name *5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
	container *
shape: *
dtype0*
_output_shapes
: 
�
)accuracy_streaming_mean/mean/count/AssignAssign"accuracy_streaming_mean/mean/count4accuracy_streaming_mean/mean/count/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
validate_shape(*
_output_shapes
: 
�
'accuracy_streaming_mean/mean/count/readIdentity"accuracy_streaming_mean/mean/count"/device:GPU:0*5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
_output_shapes
: *
T0
r
!accuracy_streaming_mean/mean/SizeConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
&accuracy_streaming_mean/mean/ToFloat_1Cast!accuracy_streaming_mean/mean/Size"/device:GPU:0*

SrcT0*
_output_shapes
: *

DstT0
t
"accuracy_streaming_mean/mean/ConstConst"/device:GPU:0*
valueB *
dtype0*
_output_shapes
: 
�
 accuracy_streaming_mean/mean/SumSumaccuracy/accuracy"accuracy_streaming_mean/mean/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
&accuracy_streaming_mean/mean/AssignAdd	AssignAdd"accuracy_streaming_mean/mean/total accuracy_streaming_mean/mean/Sum"/device:GPU:0*
use_locking( *
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
_output_shapes
: 
�
(accuracy_streaming_mean/mean/AssignAdd_1	AssignAdd"accuracy_streaming_mean/mean/count&accuracy_streaming_mean/mean/ToFloat_1^accuracy/accuracy"/device:GPU:0*5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
_output_shapes
: *
use_locking( *
T0
z
&accuracy_streaming_mean/mean/Greater/yConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$accuracy_streaming_mean/mean/GreaterGreater'accuracy_streaming_mean/mean/count/read&accuracy_streaming_mean/mean/Greater/y"/device:GPU:0*
T0*
_output_shapes
: 
�
$accuracy_streaming_mean/mean/truedivRealDiv'accuracy_streaming_mean/mean/total/read'accuracy_streaming_mean/mean/count/read"/device:GPU:0*
_output_shapes
: *
T0
x
$accuracy_streaming_mean/mean/value/eConst"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
"accuracy_streaming_mean/mean/valueSelect$accuracy_streaming_mean/mean/Greater$accuracy_streaming_mean/mean/truediv$accuracy_streaming_mean/mean/value/e"/device:GPU:0*
T0*
_output_shapes
: 
|
(accuracy_streaming_mean/mean/Greater_1/yConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
&accuracy_streaming_mean/mean/Greater_1Greater(accuracy_streaming_mean/mean/AssignAdd_1(accuracy_streaming_mean/mean/Greater_1/y"/device:GPU:0*
_output_shapes
: *
T0
�
&accuracy_streaming_mean/mean/truediv_1RealDiv&accuracy_streaming_mean/mean/AssignAdd(accuracy_streaming_mean/mean/AssignAdd_1"/device:GPU:0*
T0*
_output_shapes
: 
|
(accuracy_streaming_mean/mean/update_op/eConst"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
&accuracy_streaming_mean/mean/update_opSelect&accuracy_streaming_mean/mean/Greater_1&accuracy_streaming_mean/mean/truediv_1(accuracy_streaming_mean/mean/update_op/e"/device:GPU:0*
T0*
_output_shapes
: 
�
accuracy_streaming_mean/initNoOp*^accuracy_streaming_mean/mean/total/Assign*^accuracy_streaming_mean/mean/count/Assign"/device:GPU:0
�
accuracy_streaming_mean_1/tagsConst"/device:GPU:0*
_output_shapes
: **
value!B Baccuracy_streaming_mean_1*
dtype0
�
accuracy_streaming_mean_1ScalarSummaryaccuracy_streaming_mean_1/tags"accuracy_streaming_mean/mean/value"/device:GPU:0*
_output_shapes
: *
T0
�
Merge/MergeSummaryMergeSummaryCNN1/weights/summaries/meanCNN1/weights/summaries/stddev_1CNN1/weights/summaries/maxCNN1/weights/summaries/min CNN1/weights/summaries/histogramCNN1/biases/summaries/meanCNN1/biases/summaries/stddev_1CNN1/biases/summaries/maxCNN1/biases/summaries/minCNN1/biases/summaries/histogramCNN1/activationsCNN1/batch_normCNN2/weights/summaries/meanCNN2/weights/summaries/stddev_1CNN2/weights/summaries/maxCNN2/weights/summaries/min CNN2/weights/summaries/histogramCNN2/biases/summaries/meanCNN2/biases/summaries/stddev_1CNN2/biases/summaries/maxCNN2/biases/summaries/minCNN2/biases/summaries/histogramCNN2/activationsCNN2/batch_normCNN3/weights/summaries/meanCNN3/weights/summaries/stddev_1CNN3/weights/summaries/maxCNN3/weights/summaries/min CNN3/weights/summaries/histogramCNN3/biases/summaries/meanCNN3/biases/summaries/stddev_1CNN3/biases/summaries/maxCNN3/biases/summaries/minCNN3/biases/summaries/histogramCNN3/activationsCNN3/batch_normaccuracy_streaming_mean_1"/device:GPU:0*
N%*
_output_shapes
: ""�
trainable_variables��
v
CNN1/weights/Variable:0CNN1/weights/Variable/AssignCNN1/weights/Variable/read:02CNN1/weights/truncated_normal:0
g
CNN1/biases/Variable:0CNN1/biases/Variable/AssignCNN1/biases/Variable/read:02CNN1/biases/Const:0
�
batch_normalization/gamma:0 batch_normalization/gamma/Assign batch_normalization/gamma/read:02,batch_normalization/gamma/Initializer/ones:0
�
batch_normalization/beta:0batch_normalization/beta/Assignbatch_normalization/beta/read:02,batch_normalization/beta/Initializer/zeros:0
v
CNN2/weights/Variable:0CNN2/weights/Variable/AssignCNN2/weights/Variable/read:02CNN2/weights/truncated_normal:0
g
CNN2/biases/Variable:0CNN2/biases/Variable/AssignCNN2/biases/Variable/read:02CNN2/biases/Const:0
�
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign"batch_normalization_1/gamma/read:02.batch_normalization_1/gamma/Initializer/ones:0
�
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign!batch_normalization_1/beta/read:02.batch_normalization_1/beta/Initializer/zeros:0
v
CNN3/weights/Variable:0CNN3/weights/Variable/AssignCNN3/weights/Variable/read:02CNN3/weights/truncated_normal:0
g
CNN3/biases/Variable:0CNN3/biases/Variable/AssignCNN3/biases/Variable/read:02CNN3/biases/Const:0
�
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign"batch_normalization_2/gamma/read:02.batch_normalization_2/gamma/Initializer/ones:0
�
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign!batch_normalization_2/beta/read:02.batch_normalization_2/beta/Initializer/zeros:0
R
FC1/Variable:0FC1/Variable/AssignFC1/Variable/read:02FC1/truncated_normal:0
M
FC1/Variable_1:0FC1/Variable_1/AssignFC1/Variable_1/read:02FC1/Const:0
�
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign"batch_normalization_3/gamma/read:02.batch_normalization_3/gamma/Initializer/ones:0
�
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign!batch_normalization_3/beta/read:02.batch_normalization_3/beta/Initializer/zeros:0
b
Readout/Variable:0Readout/Variable/AssignReadout/Variable/read:02Readout/truncated_normal:0
]
Readout/Variable_1:0Readout/Variable_1/AssignReadout/Variable_1/read:02Readout/Const:0"�
	summaries�
�
CNN1/weights/summaries/mean:0
!CNN1/weights/summaries/stddev_1:0
CNN1/weights/summaries/max:0
CNN1/weights/summaries/min:0
"CNN1/weights/summaries/histogram:0
CNN1/biases/summaries/mean:0
 CNN1/biases/summaries/stddev_1:0
CNN1/biases/summaries/max:0
CNN1/biases/summaries/min:0
!CNN1/biases/summaries/histogram:0
CNN1/activations:0
CNN1/batch_norm:0
CNN2/weights/summaries/mean:0
!CNN2/weights/summaries/stddev_1:0
CNN2/weights/summaries/max:0
CNN2/weights/summaries/min:0
"CNN2/weights/summaries/histogram:0
CNN2/biases/summaries/mean:0
 CNN2/biases/summaries/stddev_1:0
CNN2/biases/summaries/max:0
CNN2/biases/summaries/min:0
!CNN2/biases/summaries/histogram:0
CNN2/activations:0
CNN2/batch_norm:0
CNN3/weights/summaries/mean:0
!CNN3/weights/summaries/stddev_1:0
CNN3/weights/summaries/max:0
CNN3/weights/summaries/min:0
"CNN3/weights/summaries/histogram:0
CNN3/biases/summaries/mean:0
 CNN3/biases/summaries/stddev_1:0
CNN3/biases/summaries/max:0
CNN3/biases/summaries/min:0
!CNN3/biases/summaries/histogram:0
CNN3/activations:0
CNN3/batch_norm:0
accuracy_streaming_mean_1:0"�
local_variables��
�
$accuracy_streaming_mean/mean/total:0)accuracy_streaming_mean/mean/total/Assign)accuracy_streaming_mean/mean/total/read:026accuracy_streaming_mean/mean/total/Initializer/zeros:0
�
$accuracy_streaming_mean/mean/count:0)accuracy_streaming_mean/mean/count/Assign)accuracy_streaming_mean/mean/count/read:026accuracy_streaming_mean/mean/count/Initializer/zeros:0"�K
	variables�K�K
v
CNN1/weights/Variable:0CNN1/weights/Variable/AssignCNN1/weights/Variable/read:02CNN1/weights/truncated_normal:0
g
CNN1/biases/Variable:0CNN1/biases/Variable/AssignCNN1/biases/Variable/read:02CNN1/biases/Const:0
�
batch_normalization/gamma:0 batch_normalization/gamma/Assign batch_normalization/gamma/read:02,batch_normalization/gamma/Initializer/ones:0
�
batch_normalization/beta:0batch_normalization/beta/Assignbatch_normalization/beta/read:02,batch_normalization/beta/Initializer/zeros:0
�
!batch_normalization/moving_mean:0&batch_normalization/moving_mean/Assign&batch_normalization/moving_mean/read:023batch_normalization/moving_mean/Initializer/zeros:0
�
%batch_normalization/moving_variance:0*batch_normalization/moving_variance/Assign*batch_normalization/moving_variance/read:026batch_normalization/moving_variance/Initializer/ones:0
v
CNN2/weights/Variable:0CNN2/weights/Variable/AssignCNN2/weights/Variable/read:02CNN2/weights/truncated_normal:0
g
CNN2/biases/Variable:0CNN2/biases/Variable/AssignCNN2/biases/Variable/read:02CNN2/biases/Const:0
�
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign"batch_normalization_1/gamma/read:02.batch_normalization_1/gamma/Initializer/ones:0
�
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign!batch_normalization_1/beta/read:02.batch_normalization_1/beta/Initializer/zeros:0
�
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign(batch_normalization_1/moving_mean/read:025batch_normalization_1/moving_mean/Initializer/zeros:0
�
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign,batch_normalization_1/moving_variance/read:028batch_normalization_1/moving_variance/Initializer/ones:0
v
CNN3/weights/Variable:0CNN3/weights/Variable/AssignCNN3/weights/Variable/read:02CNN3/weights/truncated_normal:0
g
CNN3/biases/Variable:0CNN3/biases/Variable/AssignCNN3/biases/Variable/read:02CNN3/biases/Const:0
�
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign"batch_normalization_2/gamma/read:02.batch_normalization_2/gamma/Initializer/ones:0
�
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign!batch_normalization_2/beta/read:02.batch_normalization_2/beta/Initializer/zeros:0
�
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign(batch_normalization_2/moving_mean/read:025batch_normalization_2/moving_mean/Initializer/zeros:0
�
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign,batch_normalization_2/moving_variance/read:028batch_normalization_2/moving_variance/Initializer/ones:0
R
FC1/Variable:0FC1/Variable/AssignFC1/Variable/read:02FC1/truncated_normal:0
M
FC1/Variable_1:0FC1/Variable_1/AssignFC1/Variable_1/read:02FC1/Const:0
�
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign"batch_normalization_3/gamma/read:02.batch_normalization_3/gamma/Initializer/ones:0
�
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign!batch_normalization_3/beta/read:02.batch_normalization_3/beta/Initializer/zeros:0
�
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign(batch_normalization_3/moving_mean/read:025batch_normalization_3/moving_mean/Initializer/zeros:0
�
'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign,batch_normalization_3/moving_variance/read:028batch_normalization_3/moving_variance/Initializer/ones:0
b
Readout/Variable:0Readout/Variable/AssignReadout/Variable/read:02Readout/truncated_normal:0
]
Readout/Variable_1:0Readout/Variable_1/AssignReadout/Variable_1/read:02Readout/Const:0
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0
�
CNN1/weights/Variable/Adam:0!CNN1/weights/Variable/Adam/Assign!CNN1/weights/Variable/Adam/read:02.CNN1/weights/Variable/Adam/Initializer/zeros:0
�
CNN1/weights/Variable/Adam_1:0#CNN1/weights/Variable/Adam_1/Assign#CNN1/weights/Variable/Adam_1/read:020CNN1/weights/Variable/Adam_1/Initializer/zeros:0
�
CNN1/biases/Variable/Adam:0 CNN1/biases/Variable/Adam/Assign CNN1/biases/Variable/Adam/read:02-CNN1/biases/Variable/Adam/Initializer/zeros:0
�
CNN1/biases/Variable/Adam_1:0"CNN1/biases/Variable/Adam_1/Assign"CNN1/biases/Variable/Adam_1/read:02/CNN1/biases/Variable/Adam_1/Initializer/zeros:0
�
 batch_normalization/gamma/Adam:0%batch_normalization/gamma/Adam/Assign%batch_normalization/gamma/Adam/read:022batch_normalization/gamma/Adam/Initializer/zeros:0
�
"batch_normalization/gamma/Adam_1:0'batch_normalization/gamma/Adam_1/Assign'batch_normalization/gamma/Adam_1/read:024batch_normalization/gamma/Adam_1/Initializer/zeros:0
�
batch_normalization/beta/Adam:0$batch_normalization/beta/Adam/Assign$batch_normalization/beta/Adam/read:021batch_normalization/beta/Adam/Initializer/zeros:0
�
!batch_normalization/beta/Adam_1:0&batch_normalization/beta/Adam_1/Assign&batch_normalization/beta/Adam_1/read:023batch_normalization/beta/Adam_1/Initializer/zeros:0
�
CNN2/weights/Variable/Adam:0!CNN2/weights/Variable/Adam/Assign!CNN2/weights/Variable/Adam/read:02.CNN2/weights/Variable/Adam/Initializer/zeros:0
�
CNN2/weights/Variable/Adam_1:0#CNN2/weights/Variable/Adam_1/Assign#CNN2/weights/Variable/Adam_1/read:020CNN2/weights/Variable/Adam_1/Initializer/zeros:0
�
CNN2/biases/Variable/Adam:0 CNN2/biases/Variable/Adam/Assign CNN2/biases/Variable/Adam/read:02-CNN2/biases/Variable/Adam/Initializer/zeros:0
�
CNN2/biases/Variable/Adam_1:0"CNN2/biases/Variable/Adam_1/Assign"CNN2/biases/Variable/Adam_1/read:02/CNN2/biases/Variable/Adam_1/Initializer/zeros:0
�
"batch_normalization_1/gamma/Adam:0'batch_normalization_1/gamma/Adam/Assign'batch_normalization_1/gamma/Adam/read:024batch_normalization_1/gamma/Adam/Initializer/zeros:0
�
$batch_normalization_1/gamma/Adam_1:0)batch_normalization_1/gamma/Adam_1/Assign)batch_normalization_1/gamma/Adam_1/read:026batch_normalization_1/gamma/Adam_1/Initializer/zeros:0
�
!batch_normalization_1/beta/Adam:0&batch_normalization_1/beta/Adam/Assign&batch_normalization_1/beta/Adam/read:023batch_normalization_1/beta/Adam/Initializer/zeros:0
�
#batch_normalization_1/beta/Adam_1:0(batch_normalization_1/beta/Adam_1/Assign(batch_normalization_1/beta/Adam_1/read:025batch_normalization_1/beta/Adam_1/Initializer/zeros:0
�
CNN3/weights/Variable/Adam:0!CNN3/weights/Variable/Adam/Assign!CNN3/weights/Variable/Adam/read:02.CNN3/weights/Variable/Adam/Initializer/zeros:0
�
CNN3/weights/Variable/Adam_1:0#CNN3/weights/Variable/Adam_1/Assign#CNN3/weights/Variable/Adam_1/read:020CNN3/weights/Variable/Adam_1/Initializer/zeros:0
�
CNN3/biases/Variable/Adam:0 CNN3/biases/Variable/Adam/Assign CNN3/biases/Variable/Adam/read:02-CNN3/biases/Variable/Adam/Initializer/zeros:0
�
CNN3/biases/Variable/Adam_1:0"CNN3/biases/Variable/Adam_1/Assign"CNN3/biases/Variable/Adam_1/read:02/CNN3/biases/Variable/Adam_1/Initializer/zeros:0
�
"batch_normalization_2/gamma/Adam:0'batch_normalization_2/gamma/Adam/Assign'batch_normalization_2/gamma/Adam/read:024batch_normalization_2/gamma/Adam/Initializer/zeros:0
�
$batch_normalization_2/gamma/Adam_1:0)batch_normalization_2/gamma/Adam_1/Assign)batch_normalization_2/gamma/Adam_1/read:026batch_normalization_2/gamma/Adam_1/Initializer/zeros:0
�
!batch_normalization_2/beta/Adam:0&batch_normalization_2/beta/Adam/Assign&batch_normalization_2/beta/Adam/read:023batch_normalization_2/beta/Adam/Initializer/zeros:0
�
#batch_normalization_2/beta/Adam_1:0(batch_normalization_2/beta/Adam_1/Assign(batch_normalization_2/beta/Adam_1/read:025batch_normalization_2/beta/Adam_1/Initializer/zeros:0
p
FC1/Variable/Adam:0FC1/Variable/Adam/AssignFC1/Variable/Adam/read:02%FC1/Variable/Adam/Initializer/zeros:0
x
FC1/Variable/Adam_1:0FC1/Variable/Adam_1/AssignFC1/Variable/Adam_1/read:02'FC1/Variable/Adam_1/Initializer/zeros:0
x
FC1/Variable_1/Adam:0FC1/Variable_1/Adam/AssignFC1/Variable_1/Adam/read:02'FC1/Variable_1/Adam/Initializer/zeros:0
�
FC1/Variable_1/Adam_1:0FC1/Variable_1/Adam_1/AssignFC1/Variable_1/Adam_1/read:02)FC1/Variable_1/Adam_1/Initializer/zeros:0
�
"batch_normalization_3/gamma/Adam:0'batch_normalization_3/gamma/Adam/Assign'batch_normalization_3/gamma/Adam/read:024batch_normalization_3/gamma/Adam/Initializer/zeros:0
�
$batch_normalization_3/gamma/Adam_1:0)batch_normalization_3/gamma/Adam_1/Assign)batch_normalization_3/gamma/Adam_1/read:026batch_normalization_3/gamma/Adam_1/Initializer/zeros:0
�
!batch_normalization_3/beta/Adam:0&batch_normalization_3/beta/Adam/Assign&batch_normalization_3/beta/Adam/read:023batch_normalization_3/beta/Adam/Initializer/zeros:0
�
#batch_normalization_3/beta/Adam_1:0(batch_normalization_3/beta/Adam_1/Assign(batch_normalization_3/beta/Adam_1/read:025batch_normalization_3/beta/Adam_1/Initializer/zeros:0
�
Readout/Variable/Adam:0Readout/Variable/Adam/AssignReadout/Variable/Adam/read:02)Readout/Variable/Adam/Initializer/zeros:0
�
Readout/Variable/Adam_1:0Readout/Variable/Adam_1/AssignReadout/Variable/Adam_1/read:02+Readout/Variable/Adam_1/Initializer/zeros:0
�
Readout/Variable_1/Adam:0Readout/Variable_1/Adam/AssignReadout/Variable_1/Adam/read:02+Readout/Variable_1/Adam/Initializer/zeros:0
�
Readout/Variable_1/Adam_1:0 Readout/Variable_1/Adam_1/Assign Readout/Variable_1/Adam_1/read:02-Readout/Variable_1/Adam_1/Initializer/zeros:0"
train_op


train/Adam"�9
cond_context�9�9
�
'CNN1/batch_normalization/cond/cond_text'CNN1/batch_normalization/cond/pred_id:0(CNN1/batch_normalization/cond/switch_t:0 *�
CNN1/Relu:0
%CNN1/batch_normalization/cond/Const:0
'CNN1/batch_normalization/cond/Const_1:0
5CNN1/batch_normalization/cond/FusedBatchNorm/Switch:1
7CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1:1
7CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2:1
.CNN1/batch_normalization/cond/FusedBatchNorm:0
.CNN1/batch_normalization/cond/FusedBatchNorm:1
.CNN1/batch_normalization/cond/FusedBatchNorm:2
.CNN1/batch_normalization/cond/FusedBatchNorm:3
.CNN1/batch_normalization/cond/FusedBatchNorm:4
'CNN1/batch_normalization/cond/pred_id:0
(CNN1/batch_normalization/cond/switch_t:0
batch_normalization/beta/read:0
 batch_normalization/gamma/read:0[
 batch_normalization/gamma/read:07CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1:1Z
batch_normalization/beta/read:07CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2:1D
CNN1/Relu:05CNN1/batch_normalization/cond/FusedBatchNorm/Switch:1
�

)CNN1/batch_normalization/cond/cond_text_1'CNN1/batch_normalization/cond/pred_id:0(CNN1/batch_normalization/cond/switch_f:0*�	
CNN1/Relu:0
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch:0
9CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1:0
9CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2:0
9CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_3:0
9CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4:0
0CNN1/batch_normalization/cond/FusedBatchNorm_1:0
0CNN1/batch_normalization/cond/FusedBatchNorm_1:1
0CNN1/batch_normalization/cond/FusedBatchNorm_1:2
0CNN1/batch_normalization/cond/FusedBatchNorm_1:3
0CNN1/batch_normalization/cond/FusedBatchNorm_1:4
'CNN1/batch_normalization/cond/pred_id:0
(CNN1/batch_normalization/cond/switch_f:0
batch_normalization/beta/read:0
 batch_normalization/gamma/read:0
&batch_normalization/moving_mean/read:0
*batch_normalization/moving_variance/read:0]
 batch_normalization/gamma/read:09CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1:0\
batch_normalization/beta/read:09CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2:0g
*batch_normalization/moving_variance/read:09CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4:0c
&batch_normalization/moving_mean/read:09CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_3:0F
CNN1/Relu:07CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch:0
�
'CNN2/batch_normalization/cond/cond_text'CNN2/batch_normalization/cond/pred_id:0(CNN2/batch_normalization/cond/switch_t:0 *�
CNN2/Relu:0
%CNN2/batch_normalization/cond/Const:0
'CNN2/batch_normalization/cond/Const_1:0
5CNN2/batch_normalization/cond/FusedBatchNorm/Switch:1
7CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1:1
7CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2:1
.CNN2/batch_normalization/cond/FusedBatchNorm:0
.CNN2/batch_normalization/cond/FusedBatchNorm:1
.CNN2/batch_normalization/cond/FusedBatchNorm:2
.CNN2/batch_normalization/cond/FusedBatchNorm:3
.CNN2/batch_normalization/cond/FusedBatchNorm:4
'CNN2/batch_normalization/cond/pred_id:0
(CNN2/batch_normalization/cond/switch_t:0
!batch_normalization_1/beta/read:0
"batch_normalization_1/gamma/read:0D
CNN2/Relu:05CNN2/batch_normalization/cond/FusedBatchNorm/Switch:1]
"batch_normalization_1/gamma/read:07CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1:1\
!batch_normalization_1/beta/read:07CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2:1
�

)CNN2/batch_normalization/cond/cond_text_1'CNN2/batch_normalization/cond/pred_id:0(CNN2/batch_normalization/cond/switch_f:0*�	
CNN2/Relu:0
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch:0
9CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1:0
9CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2:0
9CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_3:0
9CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4:0
0CNN2/batch_normalization/cond/FusedBatchNorm_1:0
0CNN2/batch_normalization/cond/FusedBatchNorm_1:1
0CNN2/batch_normalization/cond/FusedBatchNorm_1:2
0CNN2/batch_normalization/cond/FusedBatchNorm_1:3
0CNN2/batch_normalization/cond/FusedBatchNorm_1:4
'CNN2/batch_normalization/cond/pred_id:0
(CNN2/batch_normalization/cond/switch_f:0
!batch_normalization_1/beta/read:0
"batch_normalization_1/gamma/read:0
(batch_normalization_1/moving_mean/read:0
,batch_normalization_1/moving_variance/read:0F
CNN2/Relu:07CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch:0_
"batch_normalization_1/gamma/read:09CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1:0^
!batch_normalization_1/beta/read:09CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2:0i
,batch_normalization_1/moving_variance/read:09CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4:0e
(batch_normalization_1/moving_mean/read:09CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_3:0
�
'CNN3/batch_normalization/cond/cond_text'CNN3/batch_normalization/cond/pred_id:0(CNN3/batch_normalization/cond/switch_t:0 *�
CNN3/Relu:0
%CNN3/batch_normalization/cond/Const:0
'CNN3/batch_normalization/cond/Const_1:0
5CNN3/batch_normalization/cond/FusedBatchNorm/Switch:1
7CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1:1
7CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2:1
.CNN3/batch_normalization/cond/FusedBatchNorm:0
.CNN3/batch_normalization/cond/FusedBatchNorm:1
.CNN3/batch_normalization/cond/FusedBatchNorm:2
.CNN3/batch_normalization/cond/FusedBatchNorm:3
.CNN3/batch_normalization/cond/FusedBatchNorm:4
'CNN3/batch_normalization/cond/pred_id:0
(CNN3/batch_normalization/cond/switch_t:0
!batch_normalization_2/beta/read:0
"batch_normalization_2/gamma/read:0D
CNN3/Relu:05CNN3/batch_normalization/cond/FusedBatchNorm/Switch:1]
"batch_normalization_2/gamma/read:07CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1:1\
!batch_normalization_2/beta/read:07CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2:1
�

)CNN3/batch_normalization/cond/cond_text_1'CNN3/batch_normalization/cond/pred_id:0(CNN3/batch_normalization/cond/switch_f:0*�	
CNN3/Relu:0
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch:0
9CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1:0
9CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2:0
9CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_3:0
9CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4:0
0CNN3/batch_normalization/cond/FusedBatchNorm_1:0
0CNN3/batch_normalization/cond/FusedBatchNorm_1:1
0CNN3/batch_normalization/cond/FusedBatchNorm_1:2
0CNN3/batch_normalization/cond/FusedBatchNorm_1:3
0CNN3/batch_normalization/cond/FusedBatchNorm_1:4
'CNN3/batch_normalization/cond/pred_id:0
(CNN3/batch_normalization/cond/switch_f:0
!batch_normalization_2/beta/read:0
"batch_normalization_2/gamma/read:0
(batch_normalization_2/moving_mean/read:0
,batch_normalization_2/moving_variance/read:0i
,batch_normalization_2/moving_variance/read:09CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4:0e
(batch_normalization_2/moving_mean/read:09CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_3:0^
!batch_normalization_2/beta/read:09CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2:0F
CNN3/Relu:07CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch:0_
"batch_normalization_2/gamma/read:09CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1:0"�

update_ops�
�
*CNN1/batch_normalization/AssignMovingAvg:0
,CNN1/batch_normalization/AssignMovingAvg_1:0
*CNN2/batch_normalization/AssignMovingAvg:0
,CNN2/batch_normalization/AssignMovingAvg_1:0
*CNN3/batch_normalization/AssignMovingAvg:0
,CNN3/batch_normalization/AssignMovingAvg_1:0
)FC1/batch_normalization/AssignMovingAvg:0
+FC1/batch_normalization/AssignMovingAvg_1:0��\+      (�p	��Ļ���A*��
"
CNN1/weights/summaries/mean��Y�
&
CNN1/weights/summaries/stddev_1庸=
!
CNN1/weights/summaries/maxNBO>
!
CNN1/weights/summaries/minh�P�
�
 CNN1/weights/summaries/histogram*�	    mʿ   �I��?     ��@! �|S���)N����#@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�ܗ�SsW�<DKc��T�nK���LQ�k�1^�sO��qU���I�
����G�a�$��{E��T���C���%�V6��u�w74��7Kaa+�I�I�)�(�+A�F�&���[�?1��a˲?�[^:��"?U�4@@�$?��82?�u�w74?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @      3@      4@      4@      ;@      ?@      8@      <@      8@      >@      ?@     �B@      =@      7@      5@      5@      8@      8@      6@      0@      1@      $@      *@      $@      "@       @      &@      (@      @      @      @      @      @      @      @      @       @      �?      �?      @      �?      @      @      @              �?       @       @               @              �?               @              �?              �?              �?              �?      �?              �?              �?               @               @      �?      �?              �?      �?      @              �?      @      �?      �?       @      @       @      @      @      @      @       @      @      @       @      @       @      "@      @      (@      @      @      @      "@      $@      ,@      1@      .@      8@      5@      4@      7@      :@      ;@      @@      5@      9@      8@      ;@      :@      6@      3@      8@      4@      �?        
!
CNN1/biases/summaries/meanf��=
%
CNN1/biases/summaries/stddev_16�;
 
CNN1/biases/summaries/maxN��=
 
CNN1/biases/summaries/miny��=
�
CNN1/biases/summaries/histogram*�	    ﴷ?   ��Q�?      0@!   �L��?)@�L�-�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              �?      $@      @        
�
CNN1/activations*�   ��-q@      pA!�E�I��A)��I�&B2�
        �-���q=a�Ϭ(�>8K�ߝ�>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�*ǣ0?@��#G�'A@C\t��B@J23��D@R��'�F@t�n̫I@��`��K@���cN@Π54�P@�0n9�bR@��X�9T@
6@�F?V@r;`�xX@��iI�Z@~
�i�]@��GL:I`@߱�&�a@C��ôc@�Y��=�e@����*�g@�'�;�:j@}+��
�l@�|�6��o@�D:�tq@�������:�
           ��EcA              �?              �?              �?      �?               @               @      �?              �?       @      @      @       @       @      �?      @      �?       @              �?      @      @       @      "@      (@       @      @      @       @      ,@      2@      *@      2@      6@      0@      0@      ,@      8@      2@     �B@     �C@     �B@     �D@     �H@      N@     @P@     �P@     �H@     @S@      V@     @V@     �[@     �^@      Z@      `@     @c@      d@     @e@      i@     @j@     `m@      p@     �r@     Pu@      t@      x@     pw@     �{@     �|@     ��@     ��@     �@     h�@     ��@     X�@      �@     P�@     8�@     ��@     ��@     (�@     ��@     ��@     >�@     &�@     D�@     ��@     z�@     ��@      �@     |�@     �@     �@     �@     Ƕ@     ��@     �@     ��@     !�@    ���@     ��@    �~�@    ��@     k�@     ��@    ���@    @'�@    @#�@    ��@     H�@    @��@    �5�@     �@     r�@     x�@    `#�@     �@    �U�@    ���@     L�@    ��@    �Z�@    ���@    ���@    ��@    pM�@    ��@     =�@    �	A    �A    @�A    uA    �bA    `?A    8�A    ��	A    �A     FA    !A    A    �A    d
A    ��A    �A    p�
A    �3	A    � 
A    ��A    @�A    �h A    @��@    `��@    �f�@    Э�@     ��@    ���@    ���@    ���@    @F�@    �k�@     ܩ@     ��@     �x@        
�'
CNN1/batch_norm*�'	   �~��   ��-"@      pA!!��I��@)N2��=�oA2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
ȾG&�$��5�"�g���0�6�/n����n�����豪}0ڰ�.��fc���X$�z��K���7�>u��6
�>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@�������:�            gA     |�@    X�A   `EHbA    `�A    HAA    x�A    �A    ���@    p��@    �\�@    P��@    ���@    p��@    0��@     6�@    �x�@    ���@     ��@     ��@     ��@    ���@    ���@    @��@    ���@    ���@    �w�@    @�@    �:�@    ��@    ���@    �L�@    ��@    ��@     �@    �_�@     S�@     f�@     �@     z�@     ��@     ��@     �@     ��@     V�@     x�@     Φ@     ̣@     :�@     �@     О@     `�@     ��@     `�@     \�@     ��@     ��@     �@     8�@     `�@      �@     x�@     ��@     ��@     �~@     �~@      y@     w@     `w@     �s@     �p@     Pq@      l@      m@      g@     `e@     �b@     @a@     �`@      \@      Y@      ]@     �R@     �T@     �R@     �Q@      E@      J@     �J@      E@      N@     �E@      E@      ;@      =@      A@      =@      7@      2@      (@      2@      .@      2@      @       @      @      @      $@      "@      $@      @      @      @      @       @      @      @      @      @       @      @       @      �?      �?       @      @      @      �?              �?              �?      �?               @              �?              �?              �?      �?              �?               @              �?      �?              �?               @      �?      �?      @      �?      �?      �?      @       @       @       @       @       @       @       @      @      @      @      $@      (@      @       @      *@      0@      @      *@      4@      3@      *@      3@      ?@      :@      8@      8@     �@@      ?@      C@      E@      F@      G@      N@     �M@     @P@      O@     @P@     @Q@     @Y@     �U@     �_@     @Z@     �b@     �`@     �c@     `g@     `i@     �n@     �n@     `r@      r@     �t@     @v@     �w@     Py@     `|@     P@     (�@     ��@     ؅@     0�@     p�@     ��@     (�@     ��@     �@     x�@     ��@     T�@     t�@     ��@     ��@     ��@     ��@     .�@     ��@     h�@     .�@     {�@     ��@     ��@     E�@     �@     ��@     4�@     (�@     �@     ��@    �w�@     z�@    ��@    ���@     #�@    ��@    ���@    �A�@    �2�@     ��@    ���@    @��@     �@    @Y�@    `��@    ���@    ���@    ���@    ��@    �d�@     E�@    ��@     ��@    @E�@    0��@    ���@    ���@    0��@    �;�@    ���@    �8A    �A    �|A    xUA    H�A    ��A    8A    �tA    ��A    XiA    ��A    `0A    ���@    ���@    0��@     }�@    P��@    ���@    �U�@    L�@    ��@    ��@    ���@     �@    ���@     V�@     3�@     ��@      �@     �r@      @        
"
CNN2/weights/summaries/mean����
&
CNN2/weights/summaries/stddev_1��=
!
CNN2/weights/summaries/max��O>
!
CNN2/weights/summaries/min_$V�
�
 CNN2/weights/summaries/histogram*�	   ���ʿ   ����?      �@! �U�I{+�)Mm¾^�X@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�ji6�9���.����ڋ��vV�R9��T7����5�i}1������6�]�����[���FF�G ��ߊ4F��h���`��?�ګ>����>��~]�[�>��>M|K�>8K�ߝ�>�h���`�>����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              2@     �e@     @i@     `m@     �m@      q@     �q@     �s@     �t@     �s@     �r@     �r@     �p@     �p@     �n@     @l@     `j@     �j@     �f@     �c@     `c@     @b@     @]@     �Y@      `@     �Y@     @V@     @V@     �P@     �Q@     �N@     @Q@      J@     �D@     �A@      ?@      B@     �@@      8@      ;@      8@      1@      4@      1@      *@      3@      &@      @      *@      .@      @      "@      @      @      @      @      @      @      @      @      �?              @      @               @      �?      �?      @      �?               @       @       @      �?              @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              @       @               @              @       @      �?      �?      �?      �?      @      @       @      @      �?      @      @      @      @      @      @       @      @      @      @      *@      .@      .@      0@      ,@      ,@      1@      4@      4@      :@      9@      2@      @@      9@     �C@      A@      <@      F@      K@     �O@     �P@     �R@     �N@     �S@      W@      W@     �Z@      `@      b@     �b@     �`@      d@     `g@     �f@      k@     �k@     @m@     �o@     �o@     `r@     �r@     q@     @r@     pr@     `t@     �p@     �q@      m@     �e@      b@      &@        
!
CNN2/biases/summaries/mean���=
%
CNN2/biases/summaries/stddev_1Ѡ);
 
CNN2/biases/summaries/max��=
 
CNN2/biases/summaries/min�^�=
�
CNN2/biases/summaries/histogram*q	    ��?   �Cһ?      @@!   ;�	@)�������?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               =@      @        
�
CNN2/activations*�   �F3"@      `A!n�>�`A)xk��}&tA2�
        �-���q=[#=�؏�>K���7�>u��6
�>T�L<�>X$�z�>.��fc��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@�������:�
           ��@EA              �?              �?              �?              �?      �?      �?      �?              �?       @              �?      �?       @      �?       @      �?      �?      �?              �?       @      @      @      @       @      @      @      @      @      @      "@      $@       @      1@      "@      "@      ,@      @      &@      0@      3@      7@      =@      4@      2@      B@      A@      >@      ?@      ?@      B@      H@     �D@      L@      N@     �Q@      P@     �V@     �U@     �X@     @X@     �^@      `@      b@     �b@     �d@     �g@      h@      m@     �q@     �q@     Pr@     �r@     �v@     �x@     @{@      @     ��@     ��@      �@     x�@     x�@     ��@     ��@     �@     �@     �@     �@     ��@      �@     x�@     n�@     h�@     <�@     v�@     �@     *�@     �@     ��@     ��@     �@     ��@     ��@     ��@     Y�@     _�@     8�@     D�@    �o�@     V�@    ��@     ��@    @�@     ��@     �@    ���@     x�@    �K�@     ��@    �*�@    ���@    ���@    @��@    ���@    P�@    ��@    �>�@    ���@    ���@    ��@    `~�@    ���@     }A    
A    �A    �"
A    ��A    �A    ��A    ��A    �LA    �,A    He
A    �C
A    8Y
A     O	A    ��A    �HA    �DA    |A    p\A    �M�@    P �@    ���@    ��@    �Q�@     x�@     <�@     n�@     �h@     �J@      @        
�(
CNN2/batch_norm*�'	   ��y�   `�@      `A!p�s\	�@)-�'�6�_A2��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;
�/eq
Ⱦ����ž�XQ�þ�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d��豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���u��gr��R%�������5�L�>;9��R�>���?�ګ>����>�*��ڽ>�[�=�k�>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @�������:�            //7A    �0A    �1A    �fA    �jA    �aA    @� A     ��@    ���@    p��@     �@    pV�@    ���@    ��@    p��@    ���@    ��@     ��@     ��@    �>�@    ���@    �.�@    `A�@    �5�@    ���@    @�@    �g�@    �V�@     ��@     \�@     ��@     s�@    ���@     ��@    �M�@    �_�@    �j�@     �@     ��@     u�@     x�@     Ҵ@     K�@     װ@     �@     `�@     
�@     d�@     ��@     ��@     ��@     X�@     ؛@     ��@     ܗ@     �@     ܒ@     ��@     ȏ@     H�@     (�@     ��@     ��@     p�@     ��@     �}@     �}@     @{@     `x@     �u@     Pr@     �q@     �q@      l@     @k@     `h@     `d@     `e@     �`@     �_@     @a@      _@     �Z@     @W@     �X@     �W@     �V@     @P@      N@      D@      I@      G@      D@      ?@     �@@      =@      =@      8@      8@      0@      6@      3@      1@      .@      &@      1@      @      $@      ,@      @      &@      @      "@              $@      @      @      @      @      @      @      @      @       @      @      @      �?      �?      �?              �?               @       @               @              �?              �?              �?              �?       @              �?              �?              �?              �?              �?              �?       @               @              @       @               @       @      �?       @      @      @      @      �?      �?      @      @      @      �?      @      &@      @      @      @      @      .@       @      2@      ,@      "@      8@      2@      2@      6@      =@      2@      ,@     �@@      ?@      C@     �B@     �H@     �H@     �I@     �R@      K@     @R@      R@     �Q@      Y@     @X@      _@     �]@     @_@     @c@      d@     �h@     `g@     `k@     �n@     @m@     r@     ps@     Pw@     @y@     �x@      |@     ��@     @�@     ��@     H�@      �@     �@     ��@     �@     �@     <�@     �@     `�@     t�@     ��@     L�@     ֠@     �@     v�@     $�@     �@     �@     �@     ��@     ��@     H�@     m�@     �@     P�@     μ@     �@     &�@    ���@     ��@     ��@    ���@    ���@     c�@     B�@     ��@    ���@    ���@    ���@    @�@     7�@     ��@    @��@    ���@    @.�@     ��@    @"�@     }�@    @:�@    `��@    ���@    �K�@      �@    @��@    0��@    ���@    ���@    �
�@     o�@    ���@    @��@    p��@    �/�@    �o�@    0�@     ��@    `��@    �>�@    `��@    ���@    0p�@    �X�@    � �@    ��@     ��@     �@     �@     w�@    ���@    ���@     F�@     ��@     �n@      K@      $@      �?        
"
CNN3/weights/summaries/mean����
&
CNN3/weights/summaries/stddev_19i�=
!
CNN3/weights/summaries/maxj�Q>
!
CNN3/weights/summaries/minQS�
�
 CNN3/weights/summaries/histogram*�	   ` jʿ   @�>�?      �@! 0��y_"�)֢���o@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��f�ʜ�7
������>�?�s���O�ʗ�����Zr[v��I��P=���_�T�l�>�iD*L��>1��a˲?6�]��?����?f�ʜ�7
?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              8@     pw@     ��@     ��@     0�@     0�@     h�@     �@     `�@     ��@     �@     (�@     ��@     x�@     ��@     ��@      �@     ��@     `|@     �z@     0z@     `w@     `s@     �r@      n@     �m@     �k@      j@      g@     �c@     �c@     �c@     �`@     �\@      W@     @X@      V@     �V@     @V@     @Q@      H@     �G@      H@     �C@      F@     �A@     �B@      C@     �B@      B@      ;@      >@      5@      ;@      1@      (@      &@      (@       @      "@      @      $@       @      @       @      @      @      @       @      @       @              @       @      @      @      @       @      @      @      �?               @       @              �?              �?               @              �?              �?              �?              �?              @      �?      �?              �?       @              �?       @              @      @       @      @      @      @      @      @      "@      @      @       @      @      @      *@      &@      "@      "@      (@      1@      ,@      2@      <@      6@      8@      8@      D@     �B@     �D@      D@      G@     �B@      P@     �K@      M@     �R@     @R@     �S@      [@     @W@      ^@     �\@     �`@      b@     �f@     �f@     @k@     �m@      j@     �q@     �r@     u@     �w@     `z@     �z@      }@     �|@     �@      �@     ��@      �@     @�@     8�@     H�@     h�@     ��@     ��@     P�@     �@     ��@     �@     0@     �x@      @@        
!
CNN3/biases/summaries/meanſ�=
%
CNN3/biases/summaries/stddev_1�;
 
CNN3/biases/summaries/maxO}�=
 
CNN3/biases/summaries/min
��=
�
CNN3/biases/summaries/histogram*�	   @!ӷ?   �o�?      P@!  ���W@)VfB��?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              �?      N@      @        
�
CNN3/activations*�   ���"@      PA!8E7��|DA)�Ϲ��bVA2�	        �-���q=�XQ��>�����>;�"�q�>['�?��>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@�������:�	           ��B@A              �?              �?              �?              �?              @              �?      �?      @               @      @       @       @       @      @      @      @      @       @       @      @      @      @      @      @      $@      @      "@      (@       @      $@      (@      "@      2@      6@      ;@      7@      ;@      ;@      :@     �@@      C@      G@     �I@     �E@      O@      M@      M@     @Q@     �W@     @U@      Z@     �\@      _@     �]@     @`@     �d@     @j@     `k@     �k@     @p@     Pq@     �p@      r@     @v@      y@     �y@     p}@     �@     ؀@     ��@     H�@     P�@     0�@     ��@     ��@     Đ@     ��@     H�@     ��@     t�@     h�@     �@     b�@     ��@     L�@     
�@     �@     \�@     �@     O�@     ��@     R�@     -�@     @�@     w�@     G�@     �@     ��@    �{�@    �B�@    ���@     k�@     ��@     ��@    @3�@     ��@     ��@     Y�@    �l�@    ���@    �I�@    ���@     ��@    ���@    �D�@    @F�@    `$�@     ?�@    @4�@    ��@    ���@    ��@    ���@    �c�@    p��@    p�@    �l�@     ��@     ��@    ���@    ���@    ���@    ���@    �A�@    ��@     ��@    ��@    �z�@    ���@    ��@    ���@     +�@     ��@     �@     �@      j@      5@      @        
�$
CNN3/batch_norm*�$	   `���   �	�!@      PA!��
8�@)���:�OA2�2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ�MZ��K���u��gr����~���>�XQ��>�����>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�������:�            0��@    ��:A    ��A     P�@     ��@    ���@    ��@    ���@    @��@    @��@    ���@      �@    ���@    @��@    ���@     ��@    ���@    ���@     _�@    ���@     �@     ��@    ���@     ��@     <�@     C�@     ��@     ִ@     ��@     ��@     �@     �@     ��@     �@     r�@     �@     ��@     �@     �@     ��@     �@     <�@     ��@     ��@     L�@     @�@     x�@     �@     0�@     8�@     ��@     X�@     �~@      }@     �v@     �w@     �t@     �q@      o@     @j@     @j@     �i@      g@      d@     �e@     �^@     ``@     �]@      Y@     �U@     �U@     @R@     �P@      Q@      N@      C@     �D@      A@      =@     �B@      E@      8@      =@      :@      9@      6@      6@      1@      1@      $@       @      ,@      "@      (@       @      $@      @      @      @      �?      @      @      @      @      @      @      �?       @      �?               @      �?      @              �?              �?      �?              �?               @               @      �?               @              �?              �?              �?      �?               @              �?      �?      �?               @               @      �?               @      @      �?       @       @      @      �?      @      �?      @       @      @      @      @      @      "@       @      @      @      "@      $@      ,@      ,@      .@      &@      0@      ?@      :@      8@      7@      7@      @@      C@      F@     �A@     �M@      P@     �O@     �H@     �R@      O@      V@     @W@      ]@     @[@     �^@      `@     �f@      e@     �d@      i@     �k@      o@     �q@      r@     �v@     �u@     0z@     0y@     P}@     �~@     X�@     8�@     h�@     (�@     ��@     0�@     Џ@     @�@     �@     h�@     ̗@     ,�@     �@     L�@     �@     ��@     j�@     B�@     ��@     Ԫ@      �@     ��@     k�@     ��@     ϵ@     ��@     ۹@     "�@     5�@     �@     ��@     \�@    �{�@     W�@     ��@     V�@     d�@    �%�@     ��@    ���@    @��@    @e�@    @��@     M�@     @�@    @[�@    ���@    ���@    ���@    ��@     ��@     ��@     ��@    @��@    ��@    `Z�@    �=�@    @1�@    ��@    �,�@    @��@    �I�@    @"�@    �U�@    ���@    @r�@    @�@      �@    �)�@     6�@     �@     �@     �@     ��@      k@      D@      @        
 
accuracy_streaming_mean_1ώ~?5,KS�~      JÚ�	o�DX���A*��
"
CNN1/weights/summaries/mean�XY�
&
CNN1/weights/summaries/stddev_10��=
!
CNN1/weights/summaries/max'�O>
!
CNN1/weights/summaries/min�gP�
�
 CNN1/weights/summaries/histogram*�	    �ʿ   ����?     ��@! ���y��)TzC�>�#@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L�a�$��{E��T���C��!�A�uܬ�@8���%�V6���VlQ.��7Kaa+�1��a˲���[���S�F !?�[^:��"?uܬ�@8?��%>��:?���#@?�!�A?nK���LQ?�lDZrS?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @      3@      3@      7@      9@      <@      ;@      :@      <@      @@      >@      >@      @@      :@      4@      1@      =@      7@      4@      0@      1@      &@      &@      0@      @       @      ,@       @      @      @      @      @      @      @      @      �?      @      @              �?              @      �?              @      �?       @      @               @      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              @       @       @      @              �?      �?      �?       @      @       @       @      @       @      @      @      @      �?       @       @      @      @      (@      @      @      @      $@      @      @      @      &@      &@      *@      .@      2@      7@      4@      5@      4@      <@      ;@      =@      8@      :@      8@      8@      :@      7@      3@      9@      2@       @        
!
CNN1/biases/summaries/meanW_�=
%
CNN1/biases/summaries/stddev_1�_�;
 
CNN1/biases/summaries/max���=
 
CNN1/biases/summaries/min��=
�
CNN1/biases/summaries/histogram*�	    ���?    ��?      0@!   ��+�?)���4{�?20� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?�������:0              �?      $@      @       @        
�
CNN1/activations*�    ٣m@      pA!�
�ϦA)�&\'�[B2�
        �-���q=a�Ϭ(�>8K�ߝ�>I��P=�>��Zr[v�>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�*ǣ0?@��#G�'A@C\t��B@J23��D@R��'�F@t�n̫I@��`��K@���cN@Π54�P@�0n9�bR@��X�9T@
6@�F?V@r;`�xX@��iI�Z@~
�i�]@��GL:I`@߱�&�a@C��ôc@�Y��=�e@����*�g@�'�;�:j@}+��
�l@�|�6��o@�������:�
           @+�dA              �?              �?              �?      �?              �?       @              @              @      @      �?       @      @      �?      @      �?      @      @      @      @      @      "@      @      @      @       @      "@      @      @      $@      0@      (@      &@      1@      2@      *@      :@      ;@      B@     �B@      =@      A@     �D@     �H@     �I@      P@     �P@     �T@     @S@      T@     �S@      [@     �[@     ``@      `@     �b@     `g@      c@     �g@      k@     �j@     `m@     0q@     �r@     �s@     �v@     �y@     p{@     0�@     X�@     ��@      �@     ��@     ��@     ��@     ؍@     �@     �@     |�@     �@      �@     $�@     ܝ@     ��@     ȡ@     F�@     ��@     4�@     ^�@     ��@     r�@      �@     L�@     Q�@     $�@     �@     &�@     C�@    ���@    ���@    ���@     ��@    ���@    ���@     u�@     ��@    @X�@     �@    ���@     ��@    ���@    ���@    @D�@     �@    �M�@    �y�@    ��@    ���@    �H�@    �<�@    �3�@    ���@    ���@    p��@    @��@    �,�@    P�@    ��@    Pl�@    ���@    p�A    �A    X:A    (�A    0A    �A    wA    �A    ��A    8�A    `�A    �pA    ȘA    ��A    �A    �8A    � A    ( A    ���@    `E�@    Ps�@    0��@    0��@    � �@    �z�@    `e�@    @j�@     u�@     "�@     h�@     `�@      c@        
�&
CNN1/batch_norm*�&	   @�\�   ��$@      pA!���E�@)��0C pA2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�[�=�k���*��ڽ���������?�ګ�;9��R���'v�V,>7'_��+/>�H5�8�t>�i����v>����>豪}0ڰ>�*��ڽ>�[�=�k�>��~���>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@�������:�            ��A    ��A    $KA   �aA    �H7A    ���@     ��@     ��@    P��@    pB�@    �U�@    `�@     ��@    `_�@    �/�@    �*�@    `	�@    `X�@     G�@     �@    ���@     '�@    �`�@     ��@    ���@     l�@     
�@     [�@    �6�@     ��@     ��@    �z�@     3�@     R�@     �@     -�@     /�@     �@     �@     K�@     ޮ@     ��@     �@     ��@     ��@      �@     �@     ��@      �@     4�@     D�@     ��@     ��@      �@      �@     ��@     �@     ��@     ��@     `�@      �@     �@     z@     |@      x@     �x@     �s@     pr@      q@     `m@      l@     �c@     �d@      f@     @_@     �]@     �_@     �]@     @\@     �W@      U@     �X@     @T@     @Q@      O@     �L@      H@     �I@     �H@     �@@      =@      =@      3@      @@      5@      2@      9@      ,@      7@      4@      $@      "@      @      "@      "@      "@      @      @      @      @      @      @      @      @      @      �?      @              @      �?       @      @       @       @      @      �?               @               @              �?              �?      �?              �?              �?              �?              �?      �?              �?              @              @      �?      �?      @      �?      @      �?      @      @      �?      �?      @      @      @      @      @      $@       @      @      @      &@      @      @      0@      0@       @      *@      *@      ,@      3@      :@      5@      =@      7@      >@      A@      ;@      E@      I@      I@      K@      Q@      N@     �P@     �S@     �S@     �T@     �V@      Y@      `@      `@     �b@     `c@     �e@      j@      n@      l@     �m@     Pq@     pr@     �u@     px@     �z@     p~@     �@     h�@     h�@     X�@     `�@     ��@     8�@     ȏ@     ,�@     0�@     ��@     P�@     ��@     ��@     ��@     ��@     n�@     v�@     >�@     l�@     p�@     ��@     ��@     ��@     /�@     6�@     ��@     o�@     r�@     .�@    ���@    ���@    ���@    ��@     ��@    �<�@     �@    ���@    @�@     ��@    @��@     ��@    ��@    �[�@    `N�@     ��@    ���@     ��@    @��@    @Q�@     1�@     ��@    ���@    P��@    ��@    p��@    ��@    ���@    p�@    ���@    0 A    H� A    H�A    �)A    ��A    [A    H�A    P�A    �FA    ��A    �A    h�A    @u A    ��@    �&�@    �u�@    �i�@    ��@    `J�@    ��@    ���@    ���@    ���@    @L�@    ���@     �@     Ĥ@     ��@     �@     �{@      8@        
"
CNN2/weights/summaries/mean<u�
&
CNN2/weights/summaries/stddev_1�A�=
!
CNN2/weights/summaries/max��S>
!
CNN2/weights/summaries/minWX�
�
 CNN2/weights/summaries/histogram*�	   ��
˿   `t�?      �@! `�Ï�'�)Aq���X@2�
�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9����ڋ��vV�R9��5�i}1���d�r��h���`�8K�ߝ�;�"�q�>['�?��>6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�
              4@     �d@      i@     �m@     �n@     �p@     q@     `t@     �t@     ps@     Pr@     Ps@     �o@     �p@      m@     �k@      l@      k@     `e@      e@     @c@      _@     �`@     �Y@      `@      Z@     @Y@     �R@      Q@     �P@     �P@      N@     �H@      B@      E@      C@     �D@      :@      @@      =@      9@      7@      :@      *@      ,@      0@      (@      @      *@      @      (@      @       @      @      @      @      @      @      @      @      @       @      @      @       @      �?       @      @               @              �?       @       @              �?               @              �?              �?              �?              �?              �?              �?      �?               @               @      @      �?               @       @      �?       @       @       @               @      @      @      @      @      @      @      @      @      (@      @      "@      @      ,@      @      (@      6@      @      0@      4@      2@      7@      6@      ;@      =@      =@      :@      ;@      G@     �G@     �M@      L@     �Q@     �P@      P@      S@     @X@      Z@      V@     �`@      c@      c@     �`@     �e@     �d@     `g@     �k@     @i@     `n@     �o@      o@     `r@     r@     pr@     �q@     �r@     @t@     �p@     Pq@     �m@     �f@     �a@      ,@        
!
CNN2/biases/summaries/mean��=
%
CNN2/biases/summaries/stddev_1t�7;
 
CNN2/biases/summaries/max�N�=
 
CNN2/biases/summaries/min,��=
�
CNN2/biases/summaries/histogram*q	   �%�?   ��i�?      @@!   �3�	@)͑0�;[�?2 8/�C�ַ?%g�cE9�?��(!�ؼ?�������:               ;@      @        
�
CNN2/activations*�   ��n#@      `A!(�4_�2]A)�o���qA2�
        �-���q=;9��R�>���?�ګ>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@�������:�
            �GA               @              �?               @      �?              �?      �?               @      �?      �?              �?       @              @       @       @       @       @       @      �?      @      @      @      @              @       @       @      (@      *@      (@      @       @      *@      (@      *@      @      .@      0@      1@      3@      4@      A@      A@      6@      >@      ;@      @@     �L@      J@     �L@     �L@     �P@     �Q@     �Q@      T@     @R@     @Z@      \@     `a@     @`@     @b@     �e@     �f@      g@     �j@     �m@      o@     pr@     �u@     �v@     �y@     �{@     �}@     h�@     0�@     ��@     ��@     P�@     ��@     ��@     0�@     ��@     ��@     ��@     ��@     x�@      �@     �@     n�@     ��@     ڤ@     ��@     ��@     �@     ��@     Ű@     �@     �@     ��@     �@     �@     �@    �d�@     ��@     �@    �.�@    �f�@    ���@     9�@     ��@    ���@    @��@    @��@     ��@    @��@    �%�@     ��@     p�@    �D�@    `/�@     !�@    `��@    @*�@    ��@     ��@     2�@     �@    pF�@    ���@     7�@     *�@    �� A    ��A    P�A    0A     u	A    �_A    0CA    @�A    0�A     �A    �IA    �W
A    ��A    xTA    (�A     �A    �A    HfA    (� A    �n�@    �~�@     ��@     ��@    �'�@    @E�@     �@     ޡ@     @�@      b@      5@        
�&
CNN2/batch_norm*�&	   `e�   @� @      `A!��~�Yw�@)�YT}��_A2�������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;;�"�qʾ����ž�XQ�þ��~��¾�[�=�k��5�"�g���0�6�/n���u`P+d����n��������m!#�>�4[_>��>X$�z�>.��fc��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�������:�            7	DA    ̢A    4�A    ��A    hA    p:�@     s�@    ���@     �@    P	�@    @5�@    @��@     �@     r�@     ��@    ��@    ���@    @u�@    ���@    ���@    `Y�@    @��@    ���@     ��@    @/�@     T�@     f�@    @��@     ��@     ��@    ��@    ��@     ��@     ��@     ��@     ��@     
�@     ޹@     ��@     Ƶ@     ʲ@     ��@     ܯ@     ��@     �@     $�@     Z�@     N�@     <�@     �@     @�@     �@     \�@     D�@     В@     ��@     (�@     ��@     ��@     ��@     ��@     8�@     8�@     H�@     �|@     �}@     z@      w@     0t@      r@     �p@     �m@     @o@      n@      i@     �e@     `d@     `c@     @]@     �^@     �X@     �X@     @W@     @U@      S@     @Q@     @Q@      K@     �E@     �I@     �F@     �D@      :@      ;@      8@      6@      8@      :@      2@      9@      7@      .@      *@      "@      ,@      *@       @      @      @      @      @      @      @      @      @      @      �?      @      �?      @      @      @      @      @       @      �?      �?              �?      �?               @              �?              �?      �?       @              �?              �?               @               @      �?              @      �?      @              �?      �?      @       @      �?      @      @      �?      @      @       @       @      @       @      �?      @      "@      @      @      "@      "@      @      $@      ,@      *@      7@      1@      <@      6@      6@      0@      =@      ?@     �D@      4@     �A@     �D@     �G@      J@      O@     �Q@     @P@     �S@     �Q@     �V@      \@     �[@     @]@     �^@     `c@      d@     �f@     �f@     �h@     �o@     �n@     �r@     @p@     �t@     �z@     �z@     �{@      �@     ��@     P�@     ��@      �@     ��@     8�@     Ȏ@     $�@     `�@     p�@     ��@     ��@     (�@     ��@     2�@     \�@     ��@     ��@     >�@     Ω@     .�@     ��@     y�@     ��@     ��@     ��@     ��@     �@     ��@    ���@     1�@    �A�@    ��@    � �@     W�@    �+�@     �@    �k�@    �>�@    �!�@    @��@     ��@    �D�@    �R�@     ��@     ,�@    ���@    ���@    ��@    @P�@    ��@    �D�@    �<�@    �%�@    @��@    ���@    @,�@    �\�@    Ё�@    ���@    0��@     X�@    ���@    �2�@    ��@    @��@    ��@    `
�@    0��@    P��@    ���@    ��@    ��@    p�@     ��@    ��@    ���@    ���@    `��@    ��@    @��@     ��@     װ@     �@     ��@     @o@      P@      .@       @        
"
CNN3/weights/summaries/mean9�!�
&
CNN3/weights/summaries/stddev_1J��=
!
CNN3/weights/summaries/maxPY>
!
CNN3/weights/summaries/min2`U�
�
 CNN3/weights/summaries/histogram*�	   @�ʿ    �#�?      �@! X�gG=�)x���\�o@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��pz�w�7��a�Ϭ(���(���
�/eq
�>;�"�q�>['�?��>��~]�[�>��>M|K�>�uE����>�f����>pz�w�7�>I��P=�>�FF�G ?��[�?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              ?@     w@     (�@     �@     x�@     ؇@     H�@     ��@     ��@     ��@     ��@     X�@     X�@     ��@     �@     ��@     ��@     ��@     p|@     �y@     �z@      v@      t@     s@     �p@      l@      l@     `g@     �i@     @c@     `c@     �b@     �^@     �Y@     �_@     �W@     @V@     �T@     �Q@     �P@      P@     �K@     �K@      F@     �@@      B@      >@     �A@      :@      C@      ?@      :@      4@      ,@      $@      2@      (@      1@      $@      &@       @       @      $@      @      $@      @       @      @      @      @      @      @       @      @       @              @       @      �?               @      �?              @              �?      �?      �?      �?       @       @              �?              �?      �?               @              �?      �?              �?              �?              �?              �?               @      @       @      @      @               @      @      �?      �?      �?      �?      �?      @      @      @      @      @      @      �?      @      @       @      @      @      &@      @      @       @      @       @      .@      .@      .@      .@      8@      =@      6@      >@      4@     �D@     �A@      C@      9@     �D@     �E@     �J@      N@     @Q@     �R@     �S@     @U@     �T@     �W@     �]@     �]@     �`@      b@     �d@     �g@     �j@     �j@     �o@     0q@     s@     �t@     �v@     �x@     �|@     �}@     }@     �@     @�@     ��@     ��@     8�@     ؅@     (�@     ��@     ؈@     (�@     �@     �@     Є@     (�@     �~@     �x@      F@        
!
CNN3/biases/summaries/mean���=
%
CNN3/biases/summaries/stddev_1!�';
 
CNN3/biases/summaries/max���=
 
CNN3/biases/summaries/minˑ�=
�
CNN3/biases/summaries/histogram*�	   `9r�?   ����?      P@!   {�;@)�(�̓��?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(               @     �M@      @        
�
CNN3/activations*�    3:$@      PA!�=��:DA)�AR&HVA2�
        �-���q=X$�z�>.��fc��>�*��ڽ>�[�=�k�>K+�E���>jqs&\��>��~]�[�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@�������:�
           ���@A              �?              �?               @      �?               @               @      �?      �?       @      �?              �?               @      �?       @      @       @       @      �?      @      �?      @       @       @       @      @      @      @      @      @      &@      .@      0@      0@      1@      4@      (@      .@      5@      7@      1@      @@      ?@      D@      >@     �G@      A@     �H@     �E@     �H@     �P@      P@     �S@     �U@     @T@     �W@     �\@     @`@      b@     `c@     �b@     �c@     �f@     �j@     `n@     �p@     pq@     @s@      t@      w@     �z@      ~@     �@     x�@     h�@     �@     ��@      �@     �@     ،@     �@     �@     ��@     L�@     @�@     \�@     ��@     F�@     ԡ@     Σ@     z�@     b�@     ��@     ��@     i�@     <�@     _�@     @�@     ��@     ��@     �@     F�@     ��@    ���@    ���@    ���@    ��@    ���@    ���@    @7�@    ���@    @��@    �}�@    ���@    @��@    ���@    ���@     	�@    `��@    ���@    `��@    ���@     ��@     h�@     ��@    0��@    ��@    �L�@    ���@    �/�@     ��@    ���@    @��@    0��@    �6�@    ���@     ��@    ���@     ��@    �)�@    ���@     �@    @S�@    �n�@     ��@     r�@     �@     "�@     �@     �u@      X@      $@      �?        
�$
CNN3/batch_norm*�$	   ��
�    C�"@      PA!8d!g1�@)/%���OA2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$�����]���>�5�L�>;9��R�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>�_�T�l�>�iD*L��>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@�������:�            Y�2A    >�*A    ��A     ,�@     ��@    `��@    �I�@     g�@    @6�@    ���@    �+�@    ��@    �H�@    ���@    �$�@     �@     �@     ��@    �Z�@    �m�@     q�@    �M�@     ��@     f�@     S�@     ˵@     |�@     ˱@     װ@     ��@     �@     �@     �@      �@     ��@     ��@     H�@     T�@     h�@     P�@     |�@     ��@     ��@     h�@     ��@     �@     X�@     (�@      �@     H�@     �~@     �}@     �z@     �w@     pw@     �r@     �s@      l@     `o@      l@     @g@     `i@     �c@     �b@     �]@     �`@      [@      [@      Y@     �T@     �T@      K@      Q@     �K@     �K@      M@      B@      D@      9@      9@      <@      @@      <@      4@      9@      5@      (@      3@      &@      "@      *@      (@      1@       @      @      $@      $@      @      "@      @      @      @      @      "@       @      @      @      �?       @               @       @      �?      �?              �?              �?      �?      �?      �?              �?              �?      �?      �?              �?      �?              �?      �?              �?              �?              @      �?              �?       @              �?       @              @      �?      �?       @      �?      �?      �?       @       @      @      @      @      $@      @      $@      (@      "@      &@      3@      0@      "@      1@      (@      3@      9@      .@      3@      @@      >@      D@     �B@      K@     �O@      K@     �M@     @P@     @Q@     @U@     @R@      W@     @Y@     �Z@      a@     �`@      d@     �d@     �e@      m@     `n@      l@     �m@     �q@      t@     �u@     y@     0{@     �~@     �@     ��@     P�@      �@     X�@     �@     ��@     T�@     �@     D�@     $�@     Ė@     ��@     ��@     �@     ��@     ��@     &�@     ��@     b�@     ��@     ��@     �@     �@     N�@     x�@     ��@     �@     <�@     ��@    ��@    ���@    ���@     Q�@    �g�@     ��@    �G�@    @E�@    �S�@    ���@    ��@     *�@    ���@    ���@    @f�@     a�@    @O�@    ���@    ���@    @��@     V�@    �)�@    ���@    �%�@    �*�@    ���@    �:�@    @{�@     ��@     ��@    ���@    @��@    �%�@     ��@    ���@    @��@    �{�@    ��@    ���@     ��@     ��@     +�@     �@     h�@     ��@     0u@     �W@      <@      �?        
 
accuracy_streaming_mean_1��?"~�      թ��	Є�䕏�A*��
"
CNN1/weights/summaries/mean'�n�
&
CNN1/weights/summaries/stddev_1�̸=
!
CNN1/weights/summaries/max]�P>
!
CNN1/weights/summaries/min�S�
�
 CNN1/weights/summaries/histogram*�	   �^`ʿ   ���?     ��@! �7��y�)�ٲ�#@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�<DKc��T��lDZrS�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@���VlQ.?��bȬ�0?�T���C?a�$��{E?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @      3@      2@      6@      9@      :@      ?@      7@      =@      A@      <@     �@@      ?@      :@      6@      ,@      ;@      <@      1@      0@      (@      ,@      1@      "@      @      (@       @      &@      @      @      @      @      @      @      @      @      �?      @       @       @       @       @      �?      @      �?              �?       @      �?              @              �?              �?      �?              @              �?              �?              �?              �?      �?              �?               @              �?      �?      �?       @              @       @      @       @      @              @      @       @       @       @       @      @      @      &@      @      @      "@      $@      @      @      "@      @      @      &@      $@      0@      *@      4@      6@      3@      8@      2@      9@      ?@      <@      5@      >@      7@      4@      >@      3@      5@      9@      3@      �?        
!
CNN1/biases/summaries/meanF��=
%
CNN1/biases/summaries/stddev_1![�;
 
CNN1/biases/summaries/maxB[�=
 
CNN1/biases/summaries/mint�=
�
CNN1/biases/summaries/histogram*�	   �n\�?   @h+�?      0@!   ��W�?)�6�b��?20� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?�������:0               @      @      @      @        
�
CNN1/activations*�   �.fl@      pA!^\��PL�A)�Ȝ[ B2�
        �-���q=;�"�q�>['�?��>8K�ߝ�>�h���`�>O�ʗ��>>�?�s��>�FF�G ?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�*ǣ0?@��#G�'A@C\t��B@J23��D@R��'�F@t�n̫I@��`��K@���cN@Π54�P@�0n9�bR@��X�9T@
6@�F?V@r;`�xX@��iI�Z@~
�i�]@��GL:I`@߱�&�a@C��ôc@�Y��=�e@����*�g@�'�;�:j@}+��
�l@�������:�
           �{�dA              �?              �?              �?      �?              @              @               @      �?       @      �?       @              @      @              �?       @      @      @      @      $@       @      @      $@      @      $@       @      $@      (@      .@      *@      0@      ,@      ,@      3@      0@      1@      :@      A@      ?@      B@      G@     �E@      E@      E@      C@      P@     �R@     @R@     @S@     �V@     @X@     @Y@     �X@      b@     �a@      c@     `e@     �i@     @m@     �l@     �p@     �q@     �q@     Pt@     @u@     0y@     `x@     @|@     �@     ؁@     `�@     ��@     �@     0�@     ��@     X�@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     �@     ¦@     ֨@     V�@     j�@     ��@     �@     ͳ@     ��@     ��@     ��@     ��@     ��@    �[�@     ��@    ���@     ��@    �M�@    ��@     g�@     ��@    �C�@    ��@    �-�@    @��@     a�@     ��@    @��@     J�@    ���@    `��@    `��@    `��@    ���@    `@�@    @"�@    ���@    P.�@    0%�@     ��@    @�@    ��@    �BA    PA    hmA    h2A    `	A     A    �dA    ��A    (A    ��A    ��A    �A    ��	A    �C
A    V	A    H,A    �:A    x� A    ��A    Ȉ A    `��@    ��@    ��@     ��@    ���@    ���@    �Y�@     ��@     ��@     h�@     ̕@     H�@        
�'
CNN1/batch_norm*�'	   ��v�   `yz$@      pA!p��=�@)H���� pA2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ��~��¾�[�=�k��G&�$��5�"�g������m!#���
�%W���H5�8�t�BvŐ�r�E'�/��x>f^��`{>�MZ��K�>��|�~�>����>豪}0ڰ>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@�������:�            `�A    �yA    ]A    ��bA    �#A    ���@    �$�@    p��@    ���@    ���@    �u�@    `��@     ��@    �|�@     !�@    �'�@     O�@    ��@    @�@    ���@    �;�@     u�@    �p�@    ���@    �b�@    ���@    ���@     [�@     ��@    ���@    ���@    �"�@    ���@     �@     �@     ~�@     }�@     �@     ��@     3�@     �@     �@     P�@     ��@     ȥ@     4�@     `�@     ��@     �@     �@     x�@     ܕ@     ��@     t�@     <�@     `�@     (�@     x�@     ��@     ��@     `�@     ��@     @}@     @|@     y@     v@     �r@      s@      q@     �q@     �k@     �i@     �g@     `e@     @a@      c@     �_@     �^@     @V@     @[@     @[@      Q@     �R@     �S@     �M@      O@     �L@      J@      G@      ?@      :@     �A@      :@      1@      :@      4@      0@      5@      3@      5@      "@      4@      (@      *@      @      $@      ,@      "@      $@      @      @       @      @               @      �?       @              @       @       @      @       @      �?      �?       @       @       @               @              �?              �?              �?              �?              �?              �?              �?              �?               @              �?       @      �?      �?      �?      �?              @      �?      �?       @      �?      @       @      @      @      @      @      @      @      @      @      @      @      @      @      "@      @      @      "@       @      2@      ,@      ,@      2@      .@      2@      2@      1@      7@      ?@      7@     �@@      B@      D@     �E@     �I@     @P@      H@     �N@     �Q@      S@     �W@     �\@      ]@     @]@     �a@     �`@     �b@      g@     `h@     �j@     `k@     Pp@     �r@     �r@     �u@     �t@     �z@     �{@     �}@     ��@     0�@     �@     ��@     p�@     ��@     �@     p�@     ̒@     ��@     ��@     �@     ��@     �@     J�@     ̡@     0�@     ��@     ��@     n�@     ��@     (�@     v�@     �@     4�@     ��@     +�@     ϻ@     ��@    ���@     ��@     h�@    ���@    �4�@      �@    ���@    @��@    @2�@     �@     "�@    �u�@    ���@    ��@    @��@     Q�@    ��@    `��@    ���@     ��@    �|�@    �G�@    ���@     ��@    ���@    `
�@    =�@    `�@    ���@    �n�@    P��@    �� A    (wA    p�A    ��A    P�A    �SA    �*A    �2A    �A    �TA     ~A    ЇA    ���@    p��@    p1�@    `��@    0v�@    ��@    ��@    p��@    `��@    ���@    ���@     ��@    ���@     �@     V�@     @�@     �@     ��@      .@        
"
CNN2/weights/summaries/mean����
&
CNN2/weights/summaries/stddev_1�k�=
!
CNN2/weights/summaries/max�wW>
!
CNN2/weights/summaries/min�%\�
�
 CNN2/weights/summaries/histogram*�	   `��˿   ����?      �@! k.yp�-�)X�����X@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���d�r�x?�x��pz�w�7��})�l a�h���`�8K�ߝ���(��澢f�����_�T�l׾��>M|Kվ�u��gr�>�MZ��K�>f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              7@     �e@     �h@      m@     @p@     �o@     �q@     0u@     ps@     ps@     �s@     �r@      p@     pq@      k@      l@     �k@     �j@     `g@     �c@     @a@      c@      \@     �[@     �_@      ]@     @W@     �P@     @R@     @Q@     �O@     �G@      H@     �H@      I@      @@      B@      @@      B@      8@      6@      5@      5@      6@      *@      ,@      @      *@      ,@      @      @      @      @      "@      @      �?      @      @      $@      @      @      �?       @      @      @       @       @      �?       @              �?              @       @      @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      �?              �?      �?       @              @      �?       @       @       @       @       @      @      @       @       @      @      @      @      @      @      *@      &@      (@      &@      @       @      1@      $@      0@      2@      1@      9@      ?@      2@      ;@      ?@      8@      C@     �I@     �H@      I@     �L@     �K@      O@     �T@     �R@      V@      W@     @\@     �`@      `@      d@     �a@     �c@     �d@     @h@     �j@     �i@     �n@     �m@     �p@     �q@     �r@     �r@     `q@     `r@      t@     �p@     @r@     �k@      f@     �b@      *@        
!
CNN2/biases/summaries/mean�>�=
%
CNN2/biases/summaries/stddev_1ǷW;
 
CNN2/biases/summaries/max�x�=
 
CNN2/biases/summaries/min���=
�
CNN2/biases/summaries/histogram*�	    �~�?   @��?      @@!   �ׇ	@)i4\0d�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              �?      9@      @        
�
CNN2/activations*�   �Rl$@      `A!tEl��]A)�1;�
rA2�
        �-���q=��|�~�>���]���>��n����>�u`P+d�>5�"�g��>G&�$�>�XQ��>�����>['�?��>K+�E���>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@�������:�
            ŅGA              �?              �?               @              �?               @              @      �?              �?              �?              �?      @       @      @      @              @      @       @      @              $@      @      $@      "@       @      @      2@      &@      *@      @      *@      ,@      4@      1@      ,@      4@      .@     �A@      B@      @@      C@     �F@      F@     �H@     �G@     �N@      I@      K@     �P@     �W@     �V@     @X@      [@     �[@     �Z@     �a@     @f@     �f@     �d@     @k@     �l@     �p@     �p@     0t@      v@     Pw@     Px@     �z@     �~@     `@     ��@     ��@      �@     ȇ@     @�@     ��@     ��@     �@     ��@     ̔@     ��@     �@     ��@     ��@     T�@     �@     ��@     ��@     ��@     ��@     h�@     C�@     ��@     ��@     ��@     ��@     V�@     ,�@    ���@    ��@    �@�@     +�@    �#�@    ��@    �j�@     ��@    @��@    ���@    @��@    ���@    ��@    @��@    `��@    @H�@    `v�@     ��@    �p�@    @`�@    ���@    `��@     ��@    �P�@    �p�@    �J�@     N�@    p� A     �A     �A    ��A    �A    �
A    HA    x�A    xoA     oA    p~A     Y
A    �wA    ��A    `�A    �.A    hrA    8A    �� A    �>�@    �S�@    @��@    ���@     ��@    � �@     ��@     T�@     @�@     @W@      <@      @        
�'
CNN2/batch_norm*�'	   �U��   �2� @      `A!���D��@)Ѭb�g�_A2��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ�*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�������|�~���MZ��K���u��gr�>�MZ��K�>0�6�/n�>5�"�g��>G&�$�>�XQ��>�����>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�������:�             }�@    D�DA    ��A    ��A    �EA    XI A    P��@    �H�@    ���@    P�@     ��@    �@    p|�@     ��@    ���@     &�@     ��@     l�@    �S�@     ��@    `��@    @��@    �{�@     ��@    ��@    ���@     ��@    �.�@    @k�@    ���@    ���@     �@     C�@    �]�@     l�@     ��@     ��@     Q�@     Ÿ@     ��@     ��@     ��@     *�@     Ʈ@     �@     T�@     �@     Ȥ@     t�@     R�@     Ȟ@     h�@     �@     �@     �@     �@     x�@     ��@     �@     ��@     ��@     h�@     ؄@      �@     ؀@     �{@     �z@     py@      w@     �t@     �s@     �p@     @p@     �k@     @l@      h@     �f@     �b@     �a@      b@     �\@     �\@     @X@     �R@     �R@     �P@     �R@      L@      M@      J@      C@     �E@     �E@      G@      :@      <@      :@      2@      :@      0@      9@      &@      3@      .@      "@       @      ,@      @      "@      (@      �?      @      "@      @      @      @      @      @      @       @      �?      @      �?              @       @       @              �?      �?      �?       @               @      �?              �?      �?              �?       @              �?              �?              �?      �?              �?              �?              @      �?       @       @      �?       @      @       @       @       @      @      �?       @      �?      �?       @       @      @       @      @       @      (@      @      *@      @      "@      (@      ,@      3@      0@      6@      8@      8@      4@      4@      A@      ?@      <@      B@      D@      L@     �Q@      H@     @P@     �R@     �Q@     �V@     @X@     �[@      [@     ``@      e@     �b@     �h@      e@     @i@     �l@     @m@     �r@     `t@     �v@     �s@     �x@     �y@      |@     ؀@     ��@      �@     �@     ��@     ȉ@     ��@     p�@     ��@     ��@     @�@     ��@     ��@     0�@     j�@     �@     ��@     �@     6�@     ��@     ��@     ��@     ް@     ��@     =�@     T�@     ׸@     κ@     "�@    � �@    ��@     ��@     ��@    �>�@    �:�@    ���@    �X�@    �[�@     ��@     ��@    @��@    ���@    ���@    ���@    @�@    �z�@    �&�@     ��@    �_�@     ��@    �@�@    `��@    ���@    ���@    ���@    ��@    ���@    ���@    � �@    ���@    @��@    ���@     '�@    `��@     ��@    `>�@    ��@    @��@    �s�@    ���@    ���@    0��@    P��@    0��@    ��@     ��@     i�@    ���@    ��@     ��@    @�@    �)�@     ڲ@     �@     ��@     �b@      K@      (@      $@        
"
CNN3/weights/summaries/mean0��
&
CNN3/weights/summaries/stddev_1�=
!
CNN3/weights/summaries/maxg�[>
!
CNN3/weights/summaries/min��V�
�
 CNN3/weights/summaries/histogram*�	   ��ʿ   ��u�?      �@! ���Ltп)���D�o@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�I��P=��pz�w�7��})�l a���(��澢f���侙ѩ�-߾E��a�WܾK+�E���>jqs&\��>�iD*L��>E��a�W�>��(���>a�Ϭ(�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              A@     �v@     �@     ��@     �@     ��@     ��@     X�@     �@     �@     ��@     ��@     ؅@     ��@     P�@     `�@     ��@     ��@     P|@      |@     py@     �u@     �r@     ps@      q@      n@      h@     @h@     �i@      d@     @e@      b@     �`@      \@     �X@     @X@     �W@     �P@      M@     �P@      N@      S@      K@      E@     �@@      G@     �@@      7@     �D@      9@      9@      :@      5@      1@      3@      0@      ,@      ,@      .@      $@      &@      @      $@      @      @      @      @      @      @      @       @      "@               @      @      �?       @      @      �?       @              �?               @      @              �?      �?              �?              �?              �?              �?              �?              �?      �?      �?      �?              �?       @              �?              �?               @      �?       @      �?       @              �?       @       @              �?               @      @      @      �?      @      @      $@      @       @      @      "@      @      &@      @      "@      "@      (@      "@      &@      4@      &@      7@      (@      ?@      2@      6@      ;@      @@      A@      B@      B@     �D@      L@      L@     �I@     �K@      U@     @W@     @R@     @V@     �X@     @[@     �^@     �`@      f@     �b@      h@     `h@     `m@      m@      q@     �r@     �u@     �v@     `x@     pz@      �@     `|@     �@     ��@     ��@     �@     ��@     8�@      �@     8�@     ��@     x�@     ��@     X�@     ��@     �@     �~@     �x@      L@        
!
CNN3/biases/summaries/mean�L�=
%
CNN3/biases/summaries/stddev_1VL;
 
CNN3/biases/summaries/maxۧ�=
 
CNN3/biases/summaries/min}�=
�
CNN3/biases/summaries/histogram*�	   ��c�?   `���?      P@!  �0�	@)\-ڗ��?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      L@      @        
�
CNN3/activations*�    �G'@      PA!�0g��BA)ޕ�H�TA2�
        �-���q=5�"�g��>G&�$�>
�/eq
�>;�"�q�>jqs&\��>��~]�[�>E��a�W�>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�
            loAA              �?               @              �?              �?      �?              �?       @       @      �?              �?      @       @       @       @      �?      �?      �?      @      @      @      @      @      @      "@      @      @      @      @      ,@      (@      $@      $@      (@      ,@       @      5@      1@      6@      ;@     �D@      C@      7@      =@      E@     �F@     �F@     �H@     �U@     �R@     @Q@      W@     �Z@      X@     @[@      _@     @c@     �\@     `d@      g@     �k@     @l@     pp@     �q@     �q@     �t@     v@      v@     �y@     �}@     ��@     �@     ��@     ��@     0�@     8�@     ��@     �@     ��@     p�@     0�@     ��@     ��@     ��@     \�@     t�@     ��@     �@     ��@     ��@     �@     �@     �@     5�@     ��@     ��@     C�@     "�@     ��@     >�@     ��@    �#�@    �!�@    �Q�@    ���@     �@     6�@      �@    ���@    ���@    ���@     �@    �S�@    @��@     _�@    @��@     ��@    @V�@    ���@     �@     ��@    ���@    ���@    �A�@    @�@    ���@    `�@    �m�@    Pw�@     ��@    ��@    p��@     ��@    ���@    @�@    ���@     ��@    `��@    �O�@     ��@    ���@    �P�@    ���@     ھ@     Ҹ@     .�@     ܘ@     ��@      Q@      2@      @      �?        
�%
CNN3/batch_norm*�%	   ��   ���&@      PA!8�����@)^C[Um�OA2�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f�����uE����E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ�XQ�þ��~��¾G&�$��5�"�g����5�L�����]�������]���>�5�L�>����>豪}0ڰ>G&�$�>�*��ڽ>��~���>�XQ��>�����>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�            o�>A    LA    ���@     ��@    ���@    ���@    �\�@    ���@    @��@     h�@     ��@    @��@     0�@    ��@     S�@     ��@    �g�@     �@    ���@     �@     ��@     ��@     ��@     K�@     5�@     ��@     8�@     ��@     ��@     ��@     �@     Z�@     �@     �@     �@     D�@     ��@     \�@      �@     ��@     L�@     ��@     �@      �@     ��@     �@     H�@     `�@     (�@     �|@      @     0y@     w@     `u@     �q@     @r@      p@     �n@      f@      g@     �g@     �a@     `c@     �]@      b@      [@      U@     @X@     �S@     �U@     �Q@     �S@      M@      I@      D@     �D@      I@     �C@      @@      =@      =@      6@      2@      4@      9@      .@      ,@      &@      &@      ,@      $@      @      ,@       @       @       @       @      @      @      @      @      @      @       @       @      @      @      @              �?      �?              �?               @       @               @       @      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?              @      �?      �?      �?      �?      �?              �?      @      �?      �?              �?      @       @      @      @      @      @      @      "@      �?      @      @      "@      1@      $@      &@      $@      ,@      0@      7@      6@      5@      6@     �@@      <@      =@      9@     �D@      A@     �I@     �H@     �K@      K@      H@     �P@     �N@      W@      X@     �Y@      _@     �]@      b@     �b@     �b@      e@     �g@      j@     �m@     �m@     �q@      v@     �y@     �w@      z@     �}@     0~@     0�@     �@     h�@     ��@     p�@     ��@     x�@     �@     ܒ@     ԓ@     ԗ@     \�@     ��@     ��@     �@     \�@     
�@     ~�@     Ч@     .�@     2�@     u�@     ��@      �@     r�@     ��@     ��@     1�@     �@    ���@    �l�@    ���@     ��@     �@     `�@    ���@     ��@    �	�@    �{�@    �0�@    ���@     k�@    @_�@    �`�@    @�@     I�@    ���@     ��@    @��@    ���@    `r�@    �h�@    �e�@     _�@    @:�@     ��@    @��@    �8�@    `��@    �\�@    �k�@    �~�@    �E�@    @��@    @��@    �B�@    ���@    �$�@    � �@    � �@     :�@     ��@     \�@     X�@     X�@      a@      9@      $@       @      �?        
 
accuracy_streaming_mean_1��?�)5.�      Տ�;	���q���A*��
"
CNN1/weights/summaries/mean	b�
&
CNN1/weights/summaries/stddev_1��=
!
CNN1/weights/summaries/max1T>
!
CNN1/weights/summaries/minU5R�
�
 CNN1/weights/summaries/histogram*�	   ��Fʿ   �!��?     ��@! ��	)��)9U�
��#@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^�nK���LQ�k�1^�sO��T���C��!�A�d�\D�X=���%>��:�uܬ�@8��S�F !?�[^:��"?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @      3@      1@      8@      8@      <@      =@      7@      <@     �B@      :@      @@      @@      ;@      6@      2@      9@      :@      0@      1@      &@      &@      ,@      .@      @       @       @      (@      $@      @      @       @      �?       @      @      @      @       @      @              @      @      @      �?       @      �?              @      �?      �?              �?              �?              �?      �?              �?              �?      �?      �?              �?              �?              �?      �?              �?       @      @      @              �?      @       @       @      @      �?      @      �?      @       @      @      @      @      @      @      @      $@       @      "@       @      "@      @      "@      &@      &@      *@      ,@      5@      4@      4@      4@      6@      :@      >@      <@      6@      ;@      8@      3@      ?@      4@      4@      <@      .@      @        
!
CNN1/biases/summaries/mean���=
%
CNN1/biases/summaries/stddev_1:F$<
 
CNN1/biases/summaries/maxN,�=
 
CNN1/biases/summaries/min+c�=
�
CNN1/biases/summaries/histogram*�	   `eL�?   ����?      0@!   м�?)�6�?��?28�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?�������:8              �?      �?      @      @      @        
�
CNN1/activations*�   @��l@      pA!��dQ/�A)豁��B2�        �-���q=���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�*ǣ0?@��#G�'A@C\t��B@J23��D@R��'�F@t�n̫I@��`��K@���cN@Π54�P@�0n9�bR@��X�9T@
6@�F?V@r;`�xX@��iI�Z@~
�i�]@��GL:I`@߱�&�a@C��ôc@�Y��=�e@����*�g@�'�;�:j@}+��
�l@�������:�           �<�dA              �?               @              �?      �?              �?               @              �?       @      �?      �?       @      �?       @       @      @      �?      �?      @      @      @      @      @      �?      �?      @      @      @               @      "@      @      "@      *@       @      (@      .@      (@      .@      5@      0@      4@      :@      5@      >@      <@      <@      C@     �B@      E@     �F@      J@      P@      R@     @R@     @S@     @V@     @W@     �\@      Z@     ``@     `b@      e@      c@     `e@      g@      n@     �o@      r@     0s@      v@     �w@     `x@     �|@     �~@     p�@     ��@     X�@     ��@     ��@     �@     ��@     А@     �@     �@     `�@     p�@     �@      �@     ��@     d�@     h�@     �@      �@     �@     ��@     Ү@     �@     ޲@     T�@     ��@     ع@     ��@     ��@     ��@    ��@    ��@     ��@    ���@    ���@     ��@    ���@    �H�@    @��@    ���@    @~�@    ���@    ���@     ��@    @O�@    ���@    ���@    �d�@    �j�@    ���@     ��@     ��@    ���@    �L�@    �o�@    �m�@    ���@     H�@    �	�@    �F A    �A    �cA    h�A    (�A    `�
A    xNA    �
A    HA    8zA    �A    �r	A    x�A    x�A    pKA    ��A     �A     PA    xbA    @4 A    `B A    8 A    ���@    pK�@    ���@    @��@     y�@    ���@     f�@    ���@     ��@     p�@     l�@     p�@        
�'
CNN1/batch_norm*�&	   ���   `iN%@      pA!4e?����@)"⥨ pA2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ5�"�g���0�6�/n���5�L�����]�����5�L�>;9��R�>���?�ګ>����>��n����>�u`P+d�>0�6�/n�>5�"�g��>�[�=�k�>��~���>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@�������:�            A    �MA    �GA   @;�YA   ��KA    p A    ��@    ���@    ��@     �@    `��@    P&�@    �J�@    �}�@     ��@    `��@    �}�@    `|�@     ��@    @R�@    ���@    ���@    �k�@    @/�@    �3�@    �%�@    ���@     �@     .�@     ��@    ���@    �^�@    ���@     ��@     e�@     K�@     ��@     z�@     ʹ@     �@     �@     ��@     �@     ��@     ��@     d�@     ��@     ��@     Z�@     ܝ@      �@     8�@     ��@     8�@     �@     8�@     ��@     H�@     H�@     0�@      �@     P�@      �@     �{@     �{@     �u@     w@     �t@     �s@     �q@     `n@     �i@     �k@     �d@     @f@     �d@     @a@     @^@     �W@     �U@      \@     �X@     @T@      S@     @Q@     �O@      E@      J@     �F@      E@     �G@     �B@      8@      7@      ?@      7@      5@      8@      0@      (@      ,@      5@      *@      &@      @      "@      $@      $@      (@      @      @      @       @      @      @      @      �?      @      @      @       @       @       @       @      �?      @      �?      �?              �?      �?      �?               @              �?              �?              �?              �?              �?               @              �?      �?      @      �?      @       @      @       @              @      �?      �?       @      @      @      @      @       @      @      @      @      @      @      "@      (@      @      $@      *@      .@      0@      2@      0@      0@      2@      5@      9@     �@@      =@     �B@      E@     �C@     �E@     �E@      C@     �Q@     �M@      U@     �L@      T@     �Z@      Z@      ^@      b@     �]@     @e@     `e@     �g@     @i@     `j@     �p@     r@     �q@     @u@     �v@     �w@     @|@     x�@     ��@     ��@     x�@     ��@     X�@     H�@     ��@     @�@     d�@     �@     ��@     ��@     T�@     �@     ��@     x�@     Ң@     `�@     ��@     ��@     �@     ��@     0�@     ��@     س@     ڶ@     �@     W�@     ��@    ���@     ��@     ��@     �@    ��@     |�@     ��@    �7�@     ��@    ���@    @��@    ���@    �2�@    �'�@    �%�@    @��@     ��@     ��@     $�@    `��@     ,�@    0%�@    @g�@    ���@    `\�@    ���@    ���@    ���@     c�@     ��@     �@    0��@    Є�@    �R A    �B A    X) A    � A    �b�@    ���@    ��@    8u A    �A    ��A    �aA    @%A    ���@    ��@    �t�@    @m�@    �W�@    �G�@    `��@    ���@    @8�@     ��@    �r�@     ��@     U�@     �@     ܜ@     X�@     ��@     �b@        
"
CNN2/weights/summaries/mean�f��
&
CNN2/weights/summaries/stddev_1=��=
!
CNN2/weights/summaries/max�X[>
!
CNN2/weights/summaries/minxa�
�
 CNN2/weights/summaries/histogram*�	   � /̿   �k�?      �@! $���/�)�NR��X@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��5�i}1���d�r�x?�x���FF�G �>�?�s�����Zr[v��I��P=��a�Ϭ(�>8K�ߝ�>6�]��?����?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              �?      8@     �d@     �i@     `m@     `o@     �p@     �q@     Pu@     �s@     �q@     �t@      r@     @q@      q@      l@     @j@      m@     �j@      f@     �e@     `a@      a@     �_@     �Y@      Z@     �\@     �V@      U@     �T@     �P@      J@     �N@      F@      K@     �F@     �C@      B@      >@      >@      :@      =@      .@      4@      0@      (@      ,@      $@      (@      "@      @      @      $@       @      @      @      @      @      @       @      @      @      @       @      @       @       @              �?       @      �?       @              �?               @       @      �?               @              �?      �?              �?              �?              �?              �?               @              �?      �?              �?              �?              @      �?       @      �?              �?       @      �?      �?      @       @      �?       @       @      @      @      @       @      @      @       @       @      @      2@      4@      @      4@      (@      *@      (@      :@      6@      7@      ;@      3@     �@@     �C@      A@     �G@      I@     �G@     �N@      K@      P@      T@     �S@     �U@     �X@      \@      _@     �`@      a@     �c@      c@     @e@     �j@     �i@     �i@     �l@      p@     pp@     0q@     �s@     Pq@     pq@     �r@     �s@     �p@     @q@     `m@      f@     `b@      .@        
!
CNN2/biases/summaries/meanB�=
%
CNN2/biases/summaries/stddev_1�^};
 
CNN2/biases/summaries/max���=
 
CNN2/biases/summaries/min��=
�
CNN2/biases/summaries/histogram*�	   @�]�?   @��?      @@!   P��	@)� yc�Z�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              �?      8@      @        
�
CNN2/activations*�   �U�&@      `A!��g�^A)�iL˫�rA2�
        �-���q=��ӤP��>�
�%W�>
�}���>X$�z�>39W$:��>R%�����>�5�L�>;9��R�>0�6�/n�>5�"�g��>�*��ڽ>�[�=�k�>�XQ��>�����>;�"�q�>['�?��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�
            uGA              �?              �?              �?              �?              �?              �?               @              �?               @              @      �?              �?              @      @              @       @       @       @      @      @      @      @      @      @      @      @      @      "@      $@      ,@      "@      @      0@      .@      2@      0@      0@      ,@      4@      9@      7@      B@     �C@     �D@      <@     �D@     �G@     �I@     �K@     �W@      S@     �T@     @W@     �X@     �_@     �a@     �\@     `c@     �d@     �c@     �d@      i@     `j@     �o@      r@      v@      u@      t@     pv@     P|@      @     ��@     8�@      �@     ȃ@     ��@      �@     p�@      �@     t�@     `�@     4�@     ��@     ̙@     Ȝ@     ��@     n�@     �@     h�@     8�@     �@     ��@     ��@     ΰ@     �@     9�@     �@     ��@     ��@     ��@     O�@    ���@    ���@     ��@    ���@    ��@    ��@    ���@    �n�@    @M�@     7�@    �~�@    ���@     ~�@    @��@     ��@    `?�@    `H�@     ��@    ��@    ���@     (�@    `��@    ��@     O�@     ��@    `��@    ��@    �AA     6A    �aA     {A    p�	A    SA    �A    ��A    �A    p�A    0$A    �\A    ��	A    p�A    ȁA    ��A    �iA    H�A    ���@    �q�@    ���@    ���@    ���@     ��@     ��@     ��@     �@     �q@     �I@      "@      @        
�(
CNN2/batch_norm*�(	    ^��   @�"@      `A!�6v���@)�����_A2��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ�*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R����|�~���MZ��K���u��gr��R%������39W$:���.��fc�����ӤP��>�
�%W�>��n����>�u`P+d�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�������:�            �4,A    0~=A    h^A    L�A    Pi A    ��@    ���@    �z�@     J�@    �%�@    �y�@    ���@    �_�@    @��@    `%�@    ���@    �7�@     �@    ��@    �p�@     ��@    @e�@    ���@    @X�@    @��@     ��@    ���@    @�@    ���@    �|�@     ��@     ��@     �@     ��@    ���@     �@     0�@     +�@     ��@     <�@     �@     W�@     �@     ƭ@     �@     �@     $�@     n�@     ��@     �@     ��@     (�@     $�@     ��@     p�@     `�@     Б@     d�@     �@     ��@     �@     Ѓ@     ��@     `�@     �@     �{@      z@      y@     �u@     v@      r@      o@     �o@     `i@     �h@     �e@     @c@     �a@     �[@     �\@      X@     �Z@     �T@     �X@      U@     @P@     �O@      G@     �Q@      K@      A@     �E@      ?@      >@      8@      =@      5@      8@      3@      7@      .@      *@      (@      $@      (@      @      @      @      @       @      @      @      @      @      @      @      @       @      @      @       @       @      �?      @      �?      �?      �?      @      �?       @              @       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?       @      �?               @      @       @      @      �?      @      @      @      @       @      @      �?       @              "@      @      @       @      $@      @      @      ,@      @      @       @      (@      3@      1@      1@      1@      5@      5@      :@      >@     �A@      7@      1@      :@      >@     �J@     �F@      E@     �L@     �K@      O@     @S@      T@     �T@     �V@     �W@      `@     �a@      a@     `e@     �h@     �i@     �i@     �o@     r@      q@     pt@     �t@     pu@     �z@     �{@     ��@     Ђ@     ��@     �@     (�@     ��@     �@     Џ@      �@     H�@     �@     0�@     0�@     @�@     �@     ��@     @�@     `�@     ��@     v�@     ��@     >�@     t�@     
�@     y�@     ��@     .�@     ��@     d�@    ��@    �y�@     ]�@     ��@    �x�@     ]�@     9�@     ��@    �K�@    ���@    @��@    @}�@    ���@    @�@    ���@    @�@    ���@    @i�@    �4�@    @(�@    `�@     '�@    �p�@    ���@    P��@    �
�@    �6�@    �k�@    �^�@    �X�@    �F�@    �@�@    �[�@    P]�@    P��@    ��@    @��@    ���@    @3�@     ��@    ��@    P�@     ��@    ���@    p��@    �Q�@    `��@     ��@     m�@    ���@    ��@     ��@     ��@     �@     �@      �@     `r@     @T@      1@      @        
"
CNN3/weights/summaries/mean�@�7
&
CNN3/weights/summaries/stddev_1��=
!
CNN3/weights/summaries/max\�[>
!
CNN3/weights/summaries/minY]�
�
 CNN3/weights/summaries/histogram*�	   �"�˿   �+w�?      �@! �$(�?)����|�o@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�f�ʜ�7
������6�]���1��a˲��FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��E��a�Wܾ�iD*L�پ;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ�f����>��(���>�h���`�>�ߊ4F��>O�ʗ��>>�?�s��>�FF�G ?��[�?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�             �H@     �u@     P�@     ��@     h�@     �@     Ј@     ȇ@     8�@      �@     ؅@     ��@     `�@     ��@     @�@      �@     �@     �@     �|@     p|@     �w@     0v@     �t@     �r@      q@     �k@      i@      i@     �g@      h@     �c@     �a@      ^@      ]@      [@     @Y@     �X@     �T@      Q@     �N@     @P@     �I@      K@     �H@     �G@     �E@      8@      :@      9@      ;@      :@      7@      1@      0@      (@      *@      $@      ,@      "@      "@      @      @      &@      @      &@       @       @      @      @      @      @      @      �?      �?       @      �?      @      �?      �?      @              �?      �?      �?      �?               @              �?              @      �?              �?              �?              �?              �?              �?              �?              �?              �?              @      @               @              @      �?      �?      @       @      �?       @      @       @      @      (@      @      @      @      "@      @      @      (@      *@       @      1@      .@      1@      0@      8@      :@      :@      9@      :@      ?@     �@@     �D@      C@     �D@      J@     �J@     �P@     �N@     @S@     @Q@     �T@     �Z@     �Z@     @]@      a@      `@      c@     @c@      f@     �j@      n@     �n@     �p@     �r@     �s@     �w@     �v@      y@     h�@     0~@     h�@     @�@      �@     ��@     h�@     �@     (�@     ��@     �@     h�@      �@     �@     ��@     H�@     p@      x@     �P@        
!
CNN3/biases/summaries/meanO��=
%
CNN3/biases/summaries/stddev_1αh;
 
CNN3/biases/summaries/max�O�=
 
CNN3/biases/summaries/min�ܰ=
�
CNN3/biases/summaries/histogram*�	    ��?    �ɺ?      P@!  ����@)vhZ{�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(               @     �J@      @        
�
CNN3/activations*�   `��'@      PA!���BA)>/k��UA2�	        �-���q=�5�L�>;9��R�>�*��ڽ>�[�=�k�>��>M|K�>�_�T�l�>�ѩ�-�>���%�>�uE����>�f����>��(���>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�	            �KBA              �?              �?              �?              @              �?      @              @      �?      �?      @      @      @      @      @       @       @      @      @      @              @      @      @       @      "@      &@      @      2@      (@      $@      "@      3@      1@      2@      ;@      3@      ;@      :@      :@      9@      =@      B@     �L@     �H@      J@     �K@      O@      G@     �Q@     @S@     �T@     �W@     �Z@     �_@     �a@     `b@     �`@     @g@      g@      k@     `p@     �j@     @s@     @s@      t@     �w@     0z@     @z@     @|@     �@     ؁@      �@     �@      �@     `�@     ȍ@     ��@     ��@     �@     ��@     ��@     �@     �@      �@     $�@     �@     �@     0�@     ��@     ȫ@     .�@     	�@     {�@     Գ@     }�@     W�@     ͺ@     ��@    �)�@    ��@     ��@    �W�@    ���@     ��@     .�@    ���@    @
�@     ��@    �D�@    �$�@     T�@    @��@    ���@    @m�@     1�@    @��@    ��@     ��@    ���@    �q�@    �(�@     ��@    `��@    �[�@    0N�@    ��@    �R�@    �~�@    ���@     R�@     �@    P4�@    �C�@    ���@    ���@    �"�@    @Y�@    @��@    �#�@    �t�@    ���@     ��@     %�@     �@     ��@     p�@     p�@     �e@     �D@      @        
�$
CNN3/batch_norm*�$	   �?��   ��b&@      PA!�u����@)e<�-h�OA2�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a�h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;����ž�XQ�þ�[�=�k���*��ڽ�G&�$��豪}0ڰ��������~���>�XQ��>jqs&\��>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�            �A    X@A    ���@     w�@    �R�@    @��@    ���@    ���@    �N�@    ���@    ���@     ��@    �2�@    ��@    ���@    �6�@    ���@    ���@     ��@     �@     I�@     9�@     u�@     µ@     ��@     ��@     ��@      �@     ��@     �@     ڥ@     �@     (�@     ��@     H�@     ��@     ��@     ��@     D�@     Б@     Б@     T�@     ��@     ��@     @�@     ��@     Ѓ@     ��@     @�@     0�@      z@     y@     x@      u@     0q@     �p@     �o@     �j@      g@      f@     �e@     �]@     �\@     @^@      Z@     @U@     �U@     �T@     @S@     �O@     �J@     �D@      J@      E@     �@@      A@     �D@      @@      :@      8@      0@      5@      .@      &@      1@      *@      &@      0@      0@      (@      ,@      (@      "@      "@      @      @      �?       @      @      @      @      @       @       @      �?               @      @      @      �?              �?       @       @      @              �?      �?               @              �?              �?              �?      �?               @              �?              �?      �?              �?       @       @      �?      �?              �?      @      @      �?       @       @      �?       @      �?      @       @      @      @      @       @       @      @      "@      @              &@      $@      @      &@      $@      &@      *@      $@      0@      .@      ,@      9@      :@      =@      4@      D@     �C@      B@     �J@     �N@      K@      K@     �S@     �R@     @U@     @W@     �X@     @X@      Z@     �^@     �^@     `d@     �e@     �i@     @i@     �l@     �p@     �q@     `s@     Pt@     �w@      y@     �{@      }@     Ё@     ��@     ��@     ��@     ��@      �@     P�@     ��@     ��@     ��@     <�@     `�@     \�@     ,�@     �@     ��@     ̣@     ��@     r�@     �@     4�@     t�@     ��@     ��@     1�@     ׶@     ׸@     ��@     &�@     {�@    �q�@     ��@     ��@    ���@     
�@    �+�@    �\�@     ��@     �@     ��@    ���@    ���@    @��@    ��@    ���@    �0�@     Y�@    @��@     ��@    ���@    ���@    ���@    `��@    @S�@    ���@    @]�@    ���@    `�@    ���@    ��@    �)�@    @��@    @p�@    ���@    ��@     +�@    �\�@      �@     ��@     @�@     ��@     (�@     F�@     ��@     ��@     `k@      O@      @      �?        
 
accuracy_streaming_mean_1c�?���~      ַ1X	�������A*��
"
CNN1/weights/summaries/mean����
&
CNN1/weights/summaries/stddev_1�@�=
!
CNN1/weights/summaries/max9T>
!
CNN1/weights/summaries/min�W�
�
 CNN1/weights/summaries/histogram*�	   ���ʿ     ��?     ��@! ����)�X^��#@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�IcD���L��qU���I���VlQ.?��bȬ�0?d�\D�X=?���#@?�!�A?�T���C?IcD���L?k�1^�sO?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @      3@      3@      6@      8@      <@      =@      8@      <@     �B@      8@     �B@      <@      :@      6@      8@      5@      9@      1@      0@      &@      0@      (@      ,@      @      (@      "@      @       @      @      @      @      @      @      @      @      @      @      �?       @      @      @      �?      �?      �?      �?              �?      �?              �?      �?      �?               @              �?              �?              �?               @              �?      �?      �?      �?      @      �?       @      �?      �?      @       @       @               @      @      @      @               @      @      @      @      @      @      @      @      @      $@      $@      @      @      (@      @       @      (@      (@      0@      2@      4@      5@      6@      3@      <@      @@      7@      :@      ;@      5@      5@      =@      6@      5@      8@      0@      @        
!
CNN1/biases/summaries/mean>C�=
%
CNN1/biases/summaries/stddev_1-�J<
 
CNN1/biases/summaries/maxO'>
 
CNN1/biases/summaries/min�4�=
�
CNN1/biases/summaries/histogram*�	    �F�?   ���?      0@!   �g(�?)NFm��\�?2@�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:@              �?       @      @      @      �?       @        
�
CNN1/activations*�   @ �l@      pA!V��y�(�A)���-h B2�
        �-���q=�5�L�>;9��R�>�ߊ4F��>})�l a�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�*ǣ0?@��#G�'A@C\t��B@J23��D@R��'�F@t�n̫I@��`��K@���cN@Π54�P@�0n9�bR@��X�9T@
6@�F?V@r;`�xX@��iI�Z@~
�i�]@��GL:I`@߱�&�a@C��ôc@�Y��=�e@����*�g@�'�;�:j@}+��
�l@�|�6��o@�������:�
           @Q:eA              �?              �?              �?      �?               @               @              �?              �?      �?      @      @      @       @      @      @               @      @      @      @      @      @      @      @       @      "@      @      (@      0@      *@      $@      "@      1@      2@      9@      7@      .@      7@      5@      C@     �C@      D@     �D@      I@      O@     �K@      I@      O@     �S@      S@      W@     �Y@      `@     �X@     �`@     �c@     �d@     �d@     `l@     `h@      n@      o@     �p@     �r@     pu@     �w@     �z@     �{@     Ѐ@     @�@     ��@      �@     ��@     ��@     ��@     p�@     ��@     �@     ̓@     ��@     ��@     ș@     T�@     0�@     ��@     @�@     8�@     ��@     p�@     ��@     ��@     "�@     ��@     ;�@     ��@     �@     r�@     ��@    �U�@     ��@    �8�@     ��@     �@    �'�@    ���@     �@    ���@    @��@    �c�@    �m�@    @��@    ���@    �R�@     ��@    ��@    �@�@    �
�@    �m�@    �j�@    ���@    @G�@    ���@    ���@    ��@    ���@     z�@    ��@     ��@    P��@     V�@    ��@    �A    H�A    �QA    �A    ��A    8
A    ��A    H	A    �{A    `�A    �A    �L
A    0oA    ��A    hXA    ��A    �CA    �� A    �hA    ��@    @#�@    �3�@    @ �@    p��@     !�@    �y�@    �~�@    �Q�@     ��@     �@     ,�@     �@     �~@      �?        
�'
CNN1/batch_norm*�'	    ���   `�X'@      pA!@x���@)��S�/ pA2�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ0�6�/n���u`P+d����n�������������?�ګ��5�L�����]������z!�?��T�L<����Ő�;F��`�}6D���n����>�u`P+d�>G&�$�>�*��ڽ>��~���>�XQ��>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�            0iA    (A     ��@   @�cA    d�A    @q�@    �N�@    1�@    �t�@    `j�@    �}�@    `�@    @&�@     ��@    �E�@    �8�@    �1�@    �i�@     ��@     ��@    �d�@     )�@    @1�@    @��@    �q�@    ���@    �z�@     q�@     ��@     /�@     ��@    � �@     ��@     �@     ��@     ض@     �@     k�@     #�@     �@     �@     <�@     ��@     ��@     `�@     L�@     ��@     �@     ��@     X�@     ��@     �@     ��@     8�@     x�@     P�@      �@     ��@     0�@     (�@     �@     H�@     �|@     �z@     Pu@     �u@     �s@     0q@     �p@      l@     �e@     �j@      c@     `a@     �a@     @_@     @[@     @\@     @Y@     @X@     @U@      P@     �P@      K@     �L@     �G@     �E@      G@      B@      ?@      :@      7@      9@      6@      ;@      3@      1@      (@      .@      4@      @      4@      ,@      ,@      @      (@      @      @       @      @       @      @      @      @      @      @      @              @      �?              @      �?      �?              @      �?               @      �?      �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?      �?               @      �?      �?      @       @      @      �?       @       @      @      @      @      @      @      @      @      @      @       @      "@      ,@      1@      *@      3@      $@      ,@      $@      6@      7@      1@      @@      9@     �@@      F@      B@     �C@     �B@      E@      P@     �F@     �O@     @P@     �S@     �W@     @W@     �W@     @`@     �^@     �c@     �d@     �d@     �i@     �f@      l@     �o@     �q@      s@     pv@     �x@      x@     �|@     �@     ��@     `�@     ��@     ��@     (�@     �@     ��@     h�@     \�@     ��@     ė@     ė@     L�@     �@     0�@     ,�@     ��@     ��@     ��@     �@     Z�@     �@     E�@     $�@     �@     �@     <�@     3�@     ��@    � �@     7�@     t�@    �c�@     o�@     ��@    ���@     V�@     ��@    ���@    @��@    ���@     4�@    ��@    @v�@     K�@    � �@    ���@     ��@    ���@    ���@    �4�@    ���@    �N�@    ��@    PA�@    0��@    ���@     @�@    0��@    ���@    ���@     F�@    m�@     ��@    0� A    p�A    hSA    x�A    `OA    �!A    HA    �e�@    P��@     �@    '�@     ��@     I�@    `-�@    `h�@    P��@    �f�@    ���@     ��@    ���@    �_�@    �z�@     �@     Z�@     \�@      �@     �{@      >@        
"
CNN2/weights/summaries/mean��Ⱥ
&
CNN2/weights/summaries/stddev_1oϴ=
!
CNN2/weights/summaries/max�]]>
!
CNN2/weights/summaries/miniQg�
�
 CNN2/weights/summaries/histogram*�	    -�̿   `���?      �@! @5��3�)7$A�&�X@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��.����ڋ��T7����5�i}1�x?�x��>h�'��f�ʜ�7
��FF�G �>�?�s����f�����uE������Zr[v�>O�ʗ��>��[�?1��a˲?6�]��?����?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              �?      @@     @e@     `g@     �o@     `o@     Pp@     Pq@     �t@      u@     �q@     �t@     0r@      p@     �q@     �l@      k@     �l@     `k@     �f@      f@     @`@      d@     �V@     @\@      Z@      ^@     �W@     @T@     @S@      M@     �L@      K@      H@      L@      E@     �D@      >@     �A@     �B@     �A@     �@@      &@      *@      3@      *@      (@      1@      @       @      "@      @       @      @      @      @      @      @      @      �?      @       @      @      @      @      �?      @       @       @      @              �?      �?      @      �?      �?              �?              �?              �?      �?              �?              �?              �?              �?               @              �?              �?      �?      �?      �?      �?              �?               @      @       @       @      �?      �?      �?      @       @       @      �?      @      @      @      @      @      @       @      @      .@      $@      ,@      (@      .@      *@      &@      1@      *@      <@      8@      9@      3@      =@      H@      G@      E@      E@     �A@     �O@      L@     �R@     @T@     @V@      S@     �Y@      X@      _@     @a@     @a@      d@     �b@      e@     �i@      j@     �i@      n@      o@     0p@     0r@      r@     `r@     @q@      s@     �s@     �q@     �p@     �l@     �e@     �`@      9@        
!
CNN2/biases/summaries/mean���=
%
CNN2/biases/summaries/stddev_1՘�;
 
CNN2/biases/summaries/max��=
 
CNN2/biases/summaries/min���=
�
CNN2/biases/summaries/histogram*�	   �u�?   @�?      @@!   �U�	@)����Ȃ�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      3@      $@        
�
CNN2/activations*�    �P(@      `A!�'�a_A)�����HsA2�
        �-���q=K���7�>u��6
�>.��fc��>39W$:��>0�6�/n�>5�"�g��>�XQ��>�����>
�/eq
�>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�������:�
            �$GA              �?              �?              �?              �?       @              �?      �?      �?       @              �?      @      �?      �?      �?      @       @      �?      @       @      @       @      @      @      @      $@      @      @      &@      $@      $@      @       @      *@       @       @      (@      3@      8@      *@      3@      2@      4@      3@      =@      A@     �H@      F@      B@      B@      H@     �L@      L@     �L@     �R@     �X@     �X@      Z@     �X@      a@     @^@     �d@     `h@     �f@     �k@     @n@     �m@     �q@     `q@     Ps@     w@     �x@     @z@     �@     ��@     ��@     �@     ��@     Ȉ@     `�@     ��@     �@     ̑@     �@     4�@     ��@     ��@     ��@     ^�@     h�@     �@     `�@     ֧@     �@     4�@     ��@     ��@     ��@     *�@     ��@     ��@     ��@     O�@     ��@     y�@    �*�@     ��@    �C�@     
�@     |�@    ���@    ���@    ���@    @��@    @��@     M�@    @]�@     W�@     c�@    ���@     ��@    `��@    ���@    �5�@    p��@    ��@    P��@    ��@    �)�@    �D A    �/A     3A    hHA    HZA    
A    H�
A    P/A    �AA    ��A    �XA    8�A    UA    0�	A    ��A    � A    �]A    ��A     �A    �) A    ���@    �a�@    ��@    @�@    @��@     3�@     �@     @�@     �u@      X@      8@      @      �?        
�'
CNN2/batch_norm*�'	   @���   ��u"@      `A!��ӛ��@)��k��_A2��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ����?�ګ�;9��R��39W$:���.��fc�����z!�?��T�L<����ӤP��>�
�%W�>���m!#�>豪}0ڰ>��n����>��~���>�XQ��>�����>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@�������:�            �� A    [LAA    h�A    $<A    șA    8Y A    ���@     �@    ��@    ���@    ��@    0B�@    p��@    �@    �!�@    ��@     ��@     o�@    �y�@    ���@    ��@    `�@    ��@     ��@    ���@    ���@    @��@    ���@    @-�@    ���@    ���@     ��@     �@     #�@    �G�@     ��@     �@     ��@     �@     ��@     &�@     |�@     ��@     r�@     ��@     �@     J�@     ڤ@     d�@     2�@     �@     �@     �@     ��@     Ȕ@     ��@     T�@     ��@     h�@     ��@     ��@     ��@     ��@     ��@     �~@     0~@     �z@      v@      w@     0u@     �p@     r@     p@     �l@      l@     �e@     �`@     @d@     �^@      a@      X@     �X@     @X@     @Z@     @V@      Q@     �S@     �E@      P@      H@     �M@     �C@     �C@     �E@      1@      1@      >@      :@      2@      $@      1@      0@      0@      &@      $@      @      (@      @       @      @      @      $@      �?      @      @       @      @      @      �?      @       @      @      �?              �?      �?      @       @      @      �?      �?      �?              �?       @              �?              �?              �?              �?              �?      �?              �?              �?       @              �?              @       @      �?               @       @      �?      @      �?              @      @      @      @      @      @      @      @      $@      "@      @      @      $@      @      $@      0@      1@      *@      0@      (@      7@      <@      :@      =@      1@      <@      C@     �D@     �M@      M@     �G@     �N@     �N@     @S@     �T@     @T@     �U@     @]@     �Z@     �_@     �^@     �b@     �c@     �e@     @k@     �k@      m@     �p@     �q@     �t@     @w@     px@     �z@     p{@     �@     ��@     (�@     ��@     �@     ��@     ؎@      �@     $�@     �@     8�@     D�@     ��@     Ĝ@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     *�@     �@     4�@     �@     ��@     |�@     :�@     �@     `�@     ��@     ��@     �@    ���@    � �@     u�@    ��@    ���@    ���@    ���@    @��@    ���@    �{�@    ���@     0�@    �q�@     ��@    `��@    �$�@    �B�@    �D�@    �S�@    ���@    ���@    ��@    pc�@     ��@    @J�@    0��@    �E�@    ���@    o�@    0H�@     p�@     <�@    0��@    �t�@    �q�@    `��@    0��@    �C�@    ��@    �=�@     ��@    0��@    p��@    �\�@    @/�@    @��@    ���@    ���@     K�@     �@     �@     ��@     �u@      \@      2@      @      @        
"
CNN3/weights/summaries/mean�9
&
CNN3/weights/summaries/stddev_1��=
!
CNN3/weights/summaries/max��[>
!
CNN3/weights/summaries/min��c�
�
 CNN3/weights/summaries/histogram*�	   ��}̿   ��~�?      �@! ���8@)C,>�p@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�1��a˲���[��O�ʗ�����Zr[v��pz�w�7��})�l a�a�Ϭ(���(����uE���⾮��%ᾋh���`�>�ߊ4F��>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              �?     �L@     v@     �@     ��@     �@     ȇ@     ��@     ��@     ��@     X�@     H�@     ��@     @�@     ��@     ��@     ��@     ؁@     @~@     �@      {@     �x@      v@      s@     s@     �o@      n@     �h@     �h@     @g@     �e@     `e@     �a@     �a@     @\@     �X@     �[@     �X@      T@     �T@     �K@      L@      J@     �G@      F@     �D@     �C@      F@      =@      6@      8@      6@      7@      .@      .@      0@      $@      4@      "@      $@       @      "@      @      @      @      $@       @      @      �?      $@      @      @      @      �?      @              @      �?       @               @               @       @              �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @               @       @      �?               @      �?               @       @      @      @      @      @      @      @      @      @       @      @      (@      @      $@      (@      &@      (@      1@      $@      (@      *@      2@     �A@      4@      6@      ?@     �@@      <@     �C@     �@@     �D@      H@     �G@     @P@      Q@     �L@     �S@      V@     @U@     @T@      V@     �^@     @^@      a@     `d@     �c@     �g@      j@      m@     �k@     @r@      q@     �u@     �v@     x@      y@     �}@     P@     �@     (�@     ��@     X�@     ��@     ��@     X�@     H�@     ؈@     x�@     ��@     ��@     ��@     ��@     �~@     �w@     �S@        
!
CNN3/biases/summaries/mean`��=
%
CNN3/biases/summaries/stddev_1��;
 
CNN3/biases/summaries/max���=
 
CNN3/biases/summaries/minᐭ=
�
CNN3/biases/summaries/histogram*�	    ��?   `�պ?      P@!  �L�@)��zC&�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              *@     �H@       @        
�
CNN3/activations*�    �:*@      PA!��,Ҥ|?A)	�aGQA2�
        �-���q=.��fc��>39W$:��>R%�����>�u��gr�>;�"�q�>['�?��>K+�E���>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�������:�
             �BA              �?              �?              �?      �?              �?      �?      �?      �?              �?               @      �?       @       @      @      �?      @      @      @      @      @      @      @      @       @      �?      @      $@      $@      @      @      @       @      $@      2@      *@      5@      1@      ;@      7@      .@      :@     �@@     �C@      A@     �C@      G@     �E@      K@      P@     �N@     �Q@      P@     �S@     @U@      Z@     �_@      \@      a@     @b@      b@     �g@     �h@      l@     �l@     Pp@      r@      u@     @u@     pw@     �x@      }@     �~@     ��@     X�@     ��@     P�@     ��@     Ћ@     h�@     P�@     В@     ,�@     @�@     ,�@     ܛ@     D�@     Ġ@     �@     ޣ@     F�@      �@     ��@     T�@     }�@     >�@     Գ@     3�@     ϸ@     ��@     ��@     ¿@     ��@    �9�@     [�@     "�@     ��@     i�@    ���@    ���@     q�@    @��@     !�@    @k�@    @m�@    @K�@    @��@    ��@     ��@     C�@    `��@    ���@     N�@    ���@    �W�@    ��@    @q�@    �g�@    ���@    �B�@    `K�@    ���@    �P�@    ��@    ��@    @��@    `��@    ��@     �@     V�@    ���@    @,�@     ��@    ���@    ��@     ��@     ��@     �@     �@     ��@     `i@      I@      $@      @      �?        
�$
CNN3/batch_norm*�$	   ��   �7�*@      PA!yl?���@)X��;`PA2�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE����E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ���?�ګ�;9��R���5�L�����]����cR�k�e>:�AC)8g>;9��R�>���?�ګ>�XQ��>�����>K+�E���>jqs&\��>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�������:�            �C,A    iA8A    ��@    @w�@    ���@    �Z�@    @ �@    ���@    @i�@    �k�@    ��@    ���@    �X�@    �-�@     Z�@     �@     ��@     %�@    �w�@     ��@     ;�@     ��@     >�@     +�@     ��@     Z�@     *�@     t�@     �@     p�@     ��@     |�@     H�@     �@     ��@     �@     �@     �@     (�@     �@     (�@     (�@     X�@     �@     ��@     �@     �@     p�@     ~@     �z@     Py@     �u@     �s@     �u@     �p@      p@     @j@     �i@     @g@     �f@      a@     �`@     @`@     @\@     �T@      Y@      S@     �U@     @Q@      J@      G@      G@     �G@     �D@      A@      =@      ?@      :@      >@      7@      7@      7@      .@      2@      4@      *@      ,@      "@      $@      "@      "@      @      @      $@       @       @      @      @       @       @      �?      @      @      @      @      @      @      @      @      �?       @              �?      @      �?      �?               @               @              �?              �?              �?              �?              �?              �?               @      �?      �?      �?      �?       @              �?      @      �?      @      @      �?              @      @      @      @       @      @      @      @      @      @      &@      &@      &@      $@      "@      4@      5@      1@      "@      8@      6@      5@      <@     �@@      1@      3@      A@      A@     �E@     �F@     �J@      J@     �K@     �T@     @P@     �R@     �P@     �Z@     @\@     �\@     �a@      d@     `c@     `g@     �g@      h@     �m@     �o@      s@     �r@     0x@     �w@      z@     �{@     h�@     ��@     ��@     ��@     ��@     �@     ��@     `�@     \�@     �@     ̔@     �@     Ș@     X�@     О@     ؠ@     :�@     ��@     b�@     h�@     x�@     �@     �@     ˱@     ֳ@     ��@     ��@     �@     ��@     n�@     ��@    ���@     �@     g�@    �T�@    �j�@     ��@     �@     ��@    ���@    @��@    @��@    @��@     ��@    ��@    �x�@    ���@    ���@    `��@    ���@     �@    @$�@    `��@    �<�@    ���@     ��@    ���@    �_�@    ���@    `��@     z�@     *�@    ���@    @��@    ���@    ���@    @m�@    �j�@    �a�@    ���@     ��@     H�@     `�@     ��@     \�@     ��@     0q@     @U@      1@       @              �?        
 
accuracy_streaming_mean_1�?5yӲ�~      �YG�	B����A*��
"
CNN1/weights/summaries/mean+��
&
CNN1/weights/summaries/stddev_1�}�=
!
CNN1/weights/summaries/max�OU>
!
CNN1/weights/summaries/min�$Y�
�
 CNN1/weights/summaries/histogram*�	   @�$˿   ����?     ��@! �6�:�)���#@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[�<DKc��T��lDZrS�nK���LQ�k�1^�sO�
����G�a�$��{E����#@�d�\D�X=���%>��:���bȬ�0���VlQ.�U�4@@�$��[^:��"���%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?k�1^�sO?nK���LQ?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @      4@      3@      6@      5@      =@      >@      ;@      =@      @@      ;@      A@      >@      <@      8@      3@      <@      5@      2@      *@      &@      2@      (@      *@      @      &@       @       @      @       @      @              �?      @      @      @      @       @       @      @      @       @       @       @               @              �?      �?               @              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?      �?              �?              �?      �?      �?      @       @              �?      @       @       @       @       @      �?       @              @      @      @      @      @      $@      @      @      $@       @       @      @      $@      @      $@      *@      ,@      ,@      1@      3@      5@      6@      5@      <@      >@      7@      9@      ;@      5@      3@      @@      5@      6@      7@      0@      @        
!
CNN1/biases/summaries/mean�G�=
%
CNN1/biases/summaries/stddev_1�&V<
 
CNN1/biases/summaries/maxoc>
 
CNN1/biases/summaries/min!��=
�
CNN1/biases/summaries/histogram*�	    ķ�?   �m��?      0@!   ��h�?)�h��
��?28� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:8               @      @      @              @        
�
CNN1/activations*�   `�k@      pA!�=�AbG�A) i뤭 B2�
        �-���q=O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�*ǣ0?@��#G�'A@C\t��B@J23��D@R��'�F@t�n̫I@��`��K@���cN@Π54�P@�0n9�bR@��X�9T@
6@�F?V@r;`�xX@��iI�Z@~
�i�]@��GL:I`@߱�&�a@C��ôc@�Y��=�e@����*�g@�'�;�:j@}+��
�l@�������:�
            J�dA              @       @      �?              �?       @      �?       @      �?      �?              @       @      �?       @              �?      @      @      @      @       @      "@      @      @       @      @      $@      "@      @      0@      @       @      (@      (@       @      0@      2@      6@      ;@      2@      9@      5@      9@      B@     �E@      C@      B@      G@     �L@      F@     �P@     @S@      Q@      T@      [@      ]@     �^@     �[@     �b@      a@     �i@     �g@     @j@     �j@     �k@     �q@     Pr@      u@     y@     px@     �z@     �{@     P�@     h�@     0�@     (�@     X�@     (�@     Ȏ@     t�@     ��@     ��@     ��@     �@     h�@     ��@     �@     D�@     $�@     p�@     r�@     ��@     ��@     V�@     ذ@     .�@     )�@     ��@     ��@     �@     ��@    �=�@    ���@     �@    �)�@    �~�@    ���@    �b�@    �~�@     ��@    ��@    @��@    @��@    ���@    ���@    ���@    @R�@    �Y�@    ���@     D�@     ��@    ���@     ��@    �%�@    ��@    pV�@    ���@    Pj�@     ��@    0��@    @��@    ���@    Ц�@    �h A    ��A    ȝA    ��A    ��A    �5A    Xs	A    �d	A    x}
A    �A     �A    �&
A    �hA    ��A    X�A    H�A    ��A     �A    x�A    �;A    �A    �jA    @��@    ���@     ��@    @��@    �-�@    �;�@     �@     X�@     �@      �@     �e@        
�'
CNN1/batch_norm*�'	    ���    ��&@      pA!��#���@)�m�a��oA2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��ϾG&�$��5�"�g�����������?�ګ�
�}�����4[_>���BvŐ�r�ہkVl�p��
�%W�>���m!#�>�4[_>��>�MZ��K�>��|�~�>�5�L�>;9��R�>G&�$�>�*��ڽ>��~���>�XQ��>�����>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�            �A    0rA    @�A   @��`A    Z�9A    ��@     P�@    ���@    �C�@     0�@     R�@    �]�@     K�@    `��@    @��@    @��@    `��@    ���@    @f�@    ���@     ��@     }�@    @��@    @��@    ���@    �*�@     ��@     7�@     >�@    ���@    �'�@    ���@     -�@     ¼@     S�@     ÷@     ��@     h�@     ȱ@     ��@     �@     Ҫ@     R�@     �@     ��@     ̢@     ��@     �@     Ԛ@     p�@     ��@     ��@     L�@     |�@     �@     ��@     ؊@     x�@     ��@     ��@     �@     P@     0|@     �x@     �x@     �t@     @u@     �r@     �o@     �p@      i@     `h@     `d@     `g@     @b@     �a@     �^@     �\@     @Y@     @X@      X@     �R@      R@     �L@     �K@     �E@      J@      E@      B@     �A@      9@     �B@     �D@      7@      ,@      .@      .@      .@      .@       @      "@      &@      $@      0@      @      "@      &@      @      @      @      @       @      @      @       @      @       @      @      @      �?               @      �?      @       @      @      �?              @               @              �?              �?               @       @              �?              �?              �?              �?      �?      �?       @              @      �?       @       @      @       @      @       @      @       @       @              @      @      @      @      @      @      (@      "@       @      @      "@       @      *@      $@      "@      .@      (@      7@      1@      3@      ;@      5@     �A@      :@     �C@      D@      C@     �F@      L@     �G@     �N@     @P@     �R@     @T@     �Q@     �Z@     @Z@     �Z@     @]@     �`@     �e@     @c@     �c@     `g@     �m@      l@      o@     �q@      t@     0x@     �x@     �y@     �}@     ��@     p�@     X�@     p�@     0�@     �@     x�@     8�@     $�@     L�@     t�@     �@     �@     x�@     ��@     ��@     ��@     Ĥ@     X�@     Ψ@     ��@     ��@     �@     ѱ@     ��@     ��@     ��@     �@     �@    �#�@     ��@    �V�@    ���@     ��@     ��@     �@    �"�@    ���@     ��@     k�@     k�@    @S�@    ���@    @H�@    ���@    �e�@     ��@     ��@    ���@    ��@    ���@    ���@    �y�@    ��@    ���@    @��@    ���@    @�@    ��@    �|�@    0g�@    ��@    8� A    ,A    IA    VA    A    (c A    �� A    �:A    0WA    �LA    x�A    ��A    �j�@    @��@    ���@    P`�@    0p�@    @��@     s�@    �G�@    �%�@    @*�@    ���@     z�@     (�@     ��@     �@     ��@     @h@      0@        
"
CNN2/weights/summaries/meanS��
&
CNN2/weights/summaries/stddev_1��=
!
CNN2/weights/summaries/maxa#b>
!
CNN2/weights/summaries/minSwg�
�
 CNN2/weights/summaries/histogram*�	   `��̿    lD�?      �@! R$Ls6�)��$'3�X@2�
�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��ߊ4F��h���`[�=�k���*��ڽ�x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?�������:�
               @      A@     �e@      h@     `o@     �o@     �o@     �q@      t@     �u@     0r@     pt@     �q@     @q@     �p@     �m@     `k@     `l@     `k@     �g@     �d@     `c@     ``@      [@     @\@     �\@     �\@     �Q@     �T@      T@      P@      O@     �F@     �M@      J@      B@      E@     �D@      A@      ?@      9@      9@      8@      .@      3@      .@      1@      @      (@      @      0@       @      @      @      @      @       @      "@       @      @      @       @      �?      @       @      �?              @               @      �?      �?       @      �?              �?              �?      �?       @      �?       @              �?              �?              �?               @               @              �?              �?              �?               @       @       @      �?              �?      @      @      @              @      @      @      @      @      @      @      @      "@      &@      "@      &@      1@      $@      &@      7@      8@      9@      3@      3@      <@      =@      I@      C@     �G@     �D@     �M@     �E@     @Q@      P@     �Q@     @T@     �X@     �V@      [@     �X@     `c@     �^@     @e@      d@     �d@     �g@      j@      k@     �n@      n@     �p@     �q@     `r@     �q@     �q@     �r@     �s@     �q@     �p@     �l@     �e@     �`@      5@       @        
!
CNN2/biases/summaries/mean���=
%
CNN2/biases/summaries/stddev_1���;
 
CNN2/biases/summaries/max���=
 
CNN2/biases/summaries/minɽ�=
�
CNN2/biases/summaries/histogram*�	    �w�?   ����?      @@!   ��	@)���3ރ�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      2@      &@        
�
CNN2/activations*�    KD'@      `A!xB�BrT]A)^�h��qA2�
        �-���q=u��6
�>T�L<�>�5�L�>;9��R�>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�
            ?�IA              �?              �?              �?      �?              �?              �?      �?              �?      �?              �?       @              �?      @      @      @       @      �?      @      @       @      �?      @      @      @      @      @      @       @      "@       @      @      *@       @      .@      1@      3@      2@      .@      9@      1@      .@      :@      ;@      B@      @@     �B@      F@     �F@      I@     �L@     @S@      O@     �T@      T@     �V@     @Z@      Y@     �Y@     `a@     `a@      b@     @h@     �g@     �g@     @m@     0p@     �p@     �s@      v@     `v@     z@     0|@     p@     @�@     ��@     ��@     x�@     ��@     `�@     X�@     ��@     ��@     ��@     ��@     �@     �@     $�@     Ԡ@     ޡ@     V�@     t�@     ��@     F�@     �@     Ư@     ��@     $�@     N�@     K�@     @�@     <�@     r�@    �<�@     ��@     ��@    ���@     �@     P�@     p�@     ��@    @��@     �@    �=�@    @��@    �K�@     ��@    @O�@    ��@    `�@    @ �@     O�@    ���@    �&�@    ���@    �x�@    �x�@    ���@    �d�@    0,�@    @��@    �VA    �A    ��A    �A    P�	A    X,A    �VA    `$A    �A    hA    (�
A    
A    �A    (�A    0�A     �A    �� A    ���@    ��@    ���@    @��@     S�@     ��@     U�@     Σ@     ��@     �~@     �[@      ?@      @        
�'
CNN2/batch_norm*�'	    �j�    JI"@      `A!�ša$��@)� �W�_A2�������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾豪}0ڰ������X$�z��
�}����[#=�؏�>K���7�>u��6
�>T�L<�>39W$:��>R%�����>����>豪}0ڰ>��n����>�u`P+d�>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@�������:�            |FA    x�A    �A    ��@    P��@    �}�@    �x�@    ���@    ��@    Pm�@    ��@    �"�@    ���@    �.�@    ���@    ���@    `��@    @��@     \�@     [�@    ��@    @��@    �t�@    �0�@    @q�@    ���@    �^�@     &�@     `�@     �@    ���@     ��@    �0�@     ��@     d�@     �@     Ʒ@     ��@     u�@     ʱ@     -�@     >�@     j�@     ��@     ��@     6�@     ��@     Ġ@     ��@     ��@     H�@     ��@     �@     �@     �@     ��@     �@     ��@      �@     0�@      �@     x�@     x�@     p}@     �x@     @y@     �u@     �t@     �r@      p@     �n@      l@      h@     �f@      f@     @b@     �c@     �[@     �Z@     �Y@     �V@      U@      S@      L@      M@     �H@      L@     �J@      G@     �F@      C@     �D@      9@      >@      6@      <@      7@      3@      4@      0@      *@      @       @      $@      @      @      @      @      @      @      @      @      @      @       @       @       @       @      �?      �?      �?       @      @       @      @      �?       @      �?       @      �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?      �?               @      �?              �?      �?       @      @      �?      @              �?      @       @      �?       @       @      �?      @       @      @      @      @      @      "@      @      @      &@      @      *@       @      $@      "@      (@      .@      4@      2@      2@      ;@      >@     �B@     �A@     �G@      F@     �F@      H@      G@      L@     �P@     �T@     �S@     @U@      V@      Z@     @Z@     �[@     �`@     �a@     @g@     �e@     `j@     �i@     �l@     �p@     0p@     �r@     `v@     `x@     �x@     @{@     @~@     ��@     ��@     0�@     ��@     X�@      �@     �@     �@     ��@     �@     ܖ@     @�@     ��@     H�@     ��@     ̢@     ��@     h�@     4�@     ��@     ��@     ��@     0�@     �@     ��@     #�@     ��@     ^�@    �
�@     ��@     q�@     Z�@    �}�@     9�@     s�@     B�@    @A�@    ���@    ���@    @��@     �@    ���@    �F�@    `��@     ]�@    ���@    ���@    @��@    �=�@    `r�@    ���@    ���@    ���@    �[�@    @��@    P>�@    P|�@    0r�@     (�@    ���@    �t�@    pC�@    �F�@    ��@     ��@    �2�@    ���@    `+�@    P��@    P��@    ��@    V�@    ���@    p��@     ��@    ��@    �-�@    @��@    ���@     ��@     ;�@     ��@     �@     P�@     �c@     �H@      "@      �?        
"
CNN3/weights/summaries/mean��9
&
CNN3/weights/summaries/stddev_1N?�=
!
CNN3/weights/summaries/max-�^>
!
CNN3/weights/summaries/min�e�
�
 CNN3/weights/summaries/histogram*�	   @ �̿   ���?      �@! ���9~@)gp<U
p@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
�1��a˲���[��8K�ߝ�a�Ϭ(龋h���`�>�ߊ4F��>})�l a�>pz�w�7�>>�?�s��>�FF�G ?��[�?1��a˲?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              �?      O@     �u@     `@     H�@     @�@     x�@     8�@     8�@     �@     ��@     @�@     ��@     ��@     ��@     �@     �@     ȁ@     �@     �}@     �z@      x@      w@     �u@     Pp@     @p@     �n@      k@      h@     �g@      e@      e@      d@     @`@     �Z@     �[@     �U@     �V@     �R@     �P@      L@     @S@     �I@     �K@      F@      I@      E@      @@     �A@      =@      ,@      2@      3@      7@      (@      9@      1@      2@      (@      "@      "@      @       @      "@      @      "@      @      @      �?      @      @      @      @      @      @               @      @       @              �?      �?      �?      �?      @      �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      @      @      �?      �?      �?       @      @      �?               @       @       @      @      @      @       @      @       @       @      @      "@      @       @      @      @      $@      $@      $@      1@      2@      .@      .@      3@      0@      .@      9@      9@      >@     �@@     �D@     �A@     �A@      E@     �J@     �P@      L@      R@     @P@     �W@     @T@     @V@     @X@     @[@     @[@      a@     �c@     @g@     �f@     �i@     �h@     �p@     0q@     �q@     �t@     v@     Py@     py@     �|@     �~@     ��@     0�@      �@     h�@     ��@     �@     P�@     ��@     (�@     �@     ��@     ��@     X�@     p�@     �@     �w@      U@        
!
CNN3/biases/summaries/meand��=
%
CNN3/biases/summaries/stddev_1�m�;
 
CNN3/biases/summaries/max6~�=
 
CNN3/biases/summaries/min$v�=
�
CNN3/biases/summaries/histogram*�	   ��n�?   �Ư�?      P@!  ��L�@)*Av��?20�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:0              �?      *@      F@      @        
�
CNN3/activations*�    x'@      PA! vg93�?A)F�i�w�RA2�	        �-���q=�[�=�k�>��~���>;�"�q�>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>���%�>�uE����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�	           ��BCA              �?              �?               @               @               @              �?      @       @      �?      �?      @      �?      �?      @       @       @      @      @      @      @      @      @      $@      @      @      "@      @      &@      $@      2@      5@      *@      3@      ,@      1@      1@      6@      >@      ,@     �B@      ?@     �B@      G@     �A@     �H@      K@      N@      R@     @Q@     @P@     @U@      X@     �X@     �Z@     �a@     �b@      e@     `c@     �g@     �h@      m@     �p@     �p@     �r@      v@     Pv@     �y@     �{@     p@     ��@     ��@     ��@     h�@     ��@     (�@     ��@     0�@     ��@     L�@     ԕ@     ��@     p�@     p�@     ��@     ��@     �@     ��@     ��@     ��@     Z�@     L�@     �@     �@     ��@     ��@     Ÿ@     �@     ;@     F�@     �@     ��@    ��@     �@    ���@    ���@     ��@    ���@    ��@     `�@    @w�@    @��@    @��@    �U�@    �|�@    ���@    @_�@    ���@    @[�@    ���@    `��@    ���@    �e�@     ��@    ���@    ��@    �/�@     v�@    �N�@    �;�@     ��@    ���@    `��@    ���@    �V�@    ���@     ��@    ���@    ���@     1�@     ��@     ��@    �}�@     �@     ��@     .�@     ��@     ��@     @w@      J@      "@        
�#
CNN3/batch_norm*�#	   @�b�   `v�&@      PA!��9F%j�@)!���{PA2�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��})�l a��ߊ4F��h���`�8K�ߝ���(��澢f�����uE���⾮��%�E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;�[�=�k���*��ڽ�;9��R���5�L����n����>�u`P+d�>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�            ��A   �dAA    8FA    �q�@    `��@    ���@    @V�@     ��@     ��@    ���@     ��@    @��@    �)�@    �/�@    ��@    �f�@    �G�@     ��@     ��@     z�@     ��@     ׸@     ݶ@     �@     �@     q�@     �@     d�@     t�@     x�@     �@     ��@     �@     `�@     4�@     ��@     d�@     ��@     ��@     �@     �@     0�@     ��@      �@     ��@     �@     Ђ@     �@      @     @y@     �x@     �v@     0t@     r@      n@      m@     �g@     �j@     �d@     �c@     �`@      b@     �`@     @W@     @Y@      Z@     �V@     �Z@      R@     �S@     �J@      M@     �G@      C@     �C@     �H@      8@      9@      7@      :@      @@      ,@      4@      4@      *@      ,@      2@      0@      "@      $@      (@      @      @      &@      @      @      @      @              @               @      @      �?      �?      �?               @      �?      @               @      �?      �?               @               @      �?              �?              �?              �?              �?              �?              �?               @              �?      @               @      �?              �?      @      �?       @      �?      �?      @      @      �?      @      "@       @       @      @      @      @      @      @      $@      .@      *@      3@      .@      1@      ,@      4@      4@      8@      5@      6@      ;@      6@      ;@      >@      @@     �C@      I@     �H@     �M@     �R@     �M@      U@     @U@     @W@     @Y@     �Y@     `a@     �b@     �d@      d@     `e@     �i@      h@     0p@     �p@      s@      v@     `w@     �x@     �y@     0z@     p�@     H�@     ��@     ��@     ȅ@     h�@     ؍@     ȏ@     D�@     �@     \�@     ��@     ��@     ��@     <�@     ��@     �@     ��@     ԥ@     r�@     �@     �@     �@     ��@     ײ@     |�@     V�@     ��@     |�@     �@    ���@    ��@     �@    ���@     ��@    �^�@     w�@    ���@    ���@     ]�@    ���@    ���@     �@    ��@    @��@     ��@    @�@     >�@    �\�@    �M�@    �U�@    �p�@     ��@     ��@    �R�@     ��@    ���@     {�@    ���@     S�@    ���@     ��@    ���@    �-�@     �@     ��@    @��@     �@    ���@     ,�@    ��@     e�@     ȳ@     ��@     ��@     ԛ@     ��@     `i@      @@      @        
 
accuracy_streaming_mean_1�?�p�^�      ���	-�S'���A*π
"
CNN1/weights/summaries/meanr!��
&
CNN1/weights/summaries/stddev_17й=
!
CNN1/weights/summaries/maxz�T>
!
CNN1/weights/summaries/min��Z�
�
 CNN1/weights/summaries/histogram*�	   `:X˿   @���?     ��@! �~Sf�)Y�b��#@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^�<DKc��T��lDZrS�IcD���L��qU���I��u�w74���82���[�?1��a˲?ji6�9�?�S�F !?�[^:��"?d�\D�X=?���#@?�T���C?a�$��{E?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?�l�P�`?���%��b?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @      4@      6@      3@      <@      :@      9@      =@      >@      ?@      ?@      ?@      :@      >@      ?@      0@      9@      6@      .@      0@      *@      ,@      *@      (@      $@      &@      @       @      @      @      @      �?      �?      @      @      @      @       @      @      @      @      @      �?       @               @               @      �?      @              �?              �?              �?              �?              �?      �?              �?              �?               @              �?               @              �?              �?               @      @       @              @       @      @      @       @      @      �?      @      @      @      @      "@      @      @       @      @       @       @      @      $@      "@      (@      $@      *@      4@      0@      ;@      2@      7@      ;@      ?@      5@      :@      <@      6@      5@      :@      5@      5@      :@      .@      @        
!
CNN1/biases/summaries/mean���=
%
CNN1/biases/summaries/stddev_1�݀<
 
CNN1/biases/summaries/max�;>
 
CNN1/biases/summaries/min?�=
�
CNN1/biases/summaries/histogram*�	   �'��?   �z��?      0@!   bx�?)*�)�4�?2H�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:H              �?      @      @      @       @       @      �?        
�
CNN1/activations*�   �B,n@      pA!y_�߆�A)w=?�B2�
        �-���q=K+�E���>jqs&\��>�f����>��(���>����?f�ʜ�7
?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�*ǣ0?@��#G�'A@C\t��B@J23��D@R��'�F@t�n̫I@��`��K@���cN@Π54�P@�0n9�bR@��X�9T@
6@�F?V@r;`�xX@��iI�Z@~
�i�]@��GL:I`@߱�&�a@C��ôc@�Y��=�e@����*�g@�'�;�:j@}+��
�l@�|�6��o@�������:�
           ���eA              �?              �?              �?      �?      �?              �?      �?      �?              @      �?              @              �?      @      @       @      �?       @      @      @       @      @      @      @      @      &@       @      @      5@      &@      &@      ,@      "@      .@      6@      4@      :@      ;@      :@      >@      >@     �C@      @@      I@      K@     �J@     @Q@     �P@     �R@      P@     @W@     �X@      [@     �X@      `@      b@     @b@     �b@     �h@      h@     `j@     �m@     �r@     pr@     �s@     �v@     0w@     �y@     �|@     @�@     @�@     Ѓ@     ��@     ��@     ��@     ��@     \�@     x�@     d�@     ̔@     ��@     ��@     t�@     T�@      @     ��@     T�@     �@     P�@     z�@     b�@     ?�@     �@     ׳@     �@     j�@     ¹@     ��@     T�@    ���@     ��@     k�@    �x�@     Y�@     ��@    ���@    ��@    ���@    ���@    �2�@    @C�@    �*�@     �@    @~�@    `��@    `��@    `��@     ��@    @v�@    @!�@    ��@    4�@    ���@    @��@     ��@    �~�@    ���@    ���@    �RA    �&A    �;A    �ZA    h�A    �
A    P�A    8�A    кA    �wA    `xA    ��
A    �@A    �A    p�A    d A    ��@    �@�@    �@    p��@    0��@    p��@    ���@    @/�@    `�@     ��@     
�@     ̲@     ��@     0�@     ��@      &@        
�'
CNN1/batch_norm*�'	   �� �    zm&@      pA!�q�>2�@)�rIU�oA2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[ӾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n����n�����豪}0ڰ���������?�ګ��5�L�����]������|�~���MZ��K���u��gr�>�MZ��K�>���?�ګ>����>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�            �WA     |�@    �A    0�+A   �j�bA    X�@    ��@    ��@    ���@     ?�@    `L�@    ���@     ��@    @��@    ��@    ���@     L�@    @`�@    �P�@    ��@    ���@    ���@    �+�@    ���@     ��@     �@     ��@    ���@     I�@     G�@    ���@     ��@     �@     ��@     �@     ?�@     ̳@     "�@     ��@     2�@     ��@     `�@     ��@     R�@     ��@     ��@     ��@     h�@     ��@     ��@     ��@     X�@     ��@     ��@     ��@     ��@     x�@     @�@     �@     ��@     �@     �z@     0y@     �v@     �t@      t@     0q@      m@     @l@      i@     @h@     �d@     �a@     @a@     @Y@      \@     �Z@     @U@     �S@      S@      R@     �Q@      K@      P@     �I@      G@     �I@      9@      :@     �E@      8@      8@      9@      2@      4@      0@      0@      (@      ,@      *@      3@      @      @      @      "@      @      &@       @      @       @      @      @       @      @      @              @              @      �?              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              @      �?      �?      �?      �?               @      @              �?       @      �?      �?      @      @       @      @       @       @       @      �?      "@      @      @      @      @      @      $@      $@      $@      @      @      $@      &@      "@      (@      2@      1@      3@      1@      ;@      <@     �A@      E@      @@     �@@     �B@      J@      M@      P@     @Q@     �Q@     �R@     �S@      Y@     �X@     �X@     �_@     �^@     �b@      d@      e@     @e@     �i@     �k@     �n@     0q@      s@     �t@     �v@     z@     �~@      �@     ��@      �@     �@     ��@     H�@     ؊@     X�@     Ԑ@     ��@     �@     Ȗ@     Ș@     ԛ@     ��@     ��@     h�@     >�@     �@     ��@     `�@     &�@     C�@     e�@     y�@     +�@     ��@     ��@     O�@     ��@    �B�@    ���@     ��@     ��@     -�@    �@�@     ��@    ���@    @��@    ���@     ��@    ���@    �F�@    � �@    @��@    �?�@    ��@    �9�@    `��@    ���@     ��@    �b�@    �:�@    @�@    ��@    @��@    Px�@    @��@    ���@    �%�@    �A    �vA    �A    �A    �\A    �7A    (�A    �`A    �A    �{A    !A    �R A    �C�@    �#�@    0�@    @��@    �/�@    �H�@    @��@    ���@    ��@    `)�@     �@     ��@     ��@     Y�@     H�@     H�@     P�@     �v@      (@        
"
CNN2/weights/summaries/meanٺ
&
CNN2/weights/summaries/stddev_1s4�=
!
CNN2/weights/summaries/max�^>
!
CNN2/weights/summaries/minhlj�
�
 CNN2/weights/summaries/histogram*�	    �MͿ   ����?      �@! ���M15�)N.sm_Y@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��f�ʜ�7
������O�ʗ�����Zr[v��pz�w�7��})�l a�
�/eq
Ⱦ����ž1��a˲?6�]��?f�ʜ�7
?>h�'�?�5�i}1?�T7��?�vV�R9?��ڋ?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�               @     �F@      c@     �i@     �n@     @p@     0p@     �q@     �s@     �t@     �r@     �t@     r@     �p@      p@      l@      m@     �l@     �l@     �g@     �c@      b@     �`@     �\@      Y@      \@     �^@      U@      S@      T@     @Q@     �K@      K@     �K@      D@      C@     �G@     �D@      7@     �A@      :@      9@      (@      7@      2@      .@      6@      .@      *@      &@      @       @      @      @      @      @      @      @      @      �?      @      @              @              @              �?      @       @       @              �?               @              �?              �?      �?              �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      @       @      �?       @       @              @      �?      @       @      @      @       @       @      �?      @      @       @      @      "@      "@      (@       @      $@      (@       @      @      $@      1@      *@      0@      *@      9@      @@      ;@      @@     �A@     �D@      B@      E@     �I@      F@      I@     �P@     �Q@     @S@      T@     @R@     �Y@     �X@     �\@      `@     �a@     �b@     @e@     �d@     �e@     `k@     �k@     �l@     �p@     p@      r@     �q@     pr@     �p@     �r@     �t@     Pq@     r@      j@     `f@     `a@      4@        
!
CNN2/biases/summaries/mean�\�=
%
CNN2/biases/summaries/stddev_15�;
 
CNN2/biases/summaries/max��=
 
CNN2/biases/summaries/min�=
�
CNN2/biases/summaries/histogram*�	   `>�?   @�0�?      @@!   {��	@)Mb��ͦ�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      2@      &@        
�
CNN2/activations*�    �<(@      `A!�	��M�]A)���K�arA2�        �-���q=�u��gr�>�MZ��K�>;9��R�>���?�ګ>����>豪}0ڰ>0�6�/n�>5�"�g��>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�������:�           ��9HA              �?              �?              �?              �?              �?              @               @       @               @       @       @       @       @       @       @              @      �?      �?      @      @      �?       @      @      �?      @       @      @      @      @      &@      @      @      @      @      (@      &@      "@      @      0@      $@      0@      9@      :@      <@      5@      :@      9@      @@      C@     �G@      F@     �L@      I@      L@     �N@     @V@     @R@     @Z@     @Z@     �X@      ]@     �]@     �b@     �b@      f@     �h@     @j@     `j@     �p@     pr@      u@     �u@     �y@      x@     �|@     �@     8�@     �@     p�@     �@      �@     ��@     H�@     ��@     �@     X�@     ��@     T�@     �@     \�@     ��@     ��@     r�@     �@     p�@     ��@     ,�@     ̮@     ��@     �@     �@     ��@     �@     !�@     ��@     �@     ��@     �@    ���@    �8�@    �"�@     ��@     ��@    ���@    ���@    ���@    @��@    @��@    �x�@    ���@    �|�@    ���@    @*�@     ��@    ���@    0��@    �g�@    �a�@    �Z�@    И�@    ���@    p��@     �@     � A    ��A    hbA    ��A    p4A    8A     �
A    ��A    ��A     AA    �\A    )
A    h�A    8A    @�A    �DA    ��A    ��@    и�@    p��@    0��@     I�@    ��@    ���@    ���@     P�@     ��@     ��@     �c@     �N@      &@      @        
�'
CNN2/batch_norm*�&	   ��F�   ��i"@      `A!4џuJ�@))�ב�_A2�������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾G&�$��5�"�g���豪}0ڰ�������5�L�����]������|�~�������~�f^��`{�=�.^ol�ڿ�ɓ�i�5�"�g��>G&�$�>�XQ��>�����>
�/eq
�>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@�������:�            NBA    �*A    �A    PmA    xNA     ��@    ��@    �U�@    `��@    �U�@    �&�@    ��@    ��@    ���@     ��@    ���@    �7�@    �r�@    ���@    @2�@    @!�@    @��@    �K�@    �F�@    @z�@    ���@    �%�@    ���@    ���@     n�@    �S�@    ���@     ��@     @�@     �@     �@     ��@     ��@     �@     G�@     į@     "�@     �@     �@     d�@     �@     j�@     �@      �@     |�@     �@     x�@      �@     ,�@     �@     ��@     h�@     ��@     �@     ��@     ��@     ��@     p�@      }@      |@     0v@     0u@     �s@     Pr@     `p@      p@     @n@     �g@     �g@      d@      `@     @`@     @[@     �[@      X@     �T@     �R@     @T@      S@     �O@     �J@     �H@     �G@      C@     �C@      @@      A@      9@      B@      @@      0@      9@      1@      8@      *@      1@      ,@      (@      .@      "@      "@       @      &@       @      @      @       @      @      @      @      @      @      �?      @      @       @      @              �?      �?              �?      �?      �?      �?              �?      �?      �?              �?              �?              �?      �?              �?              �?              �?               @      �?              �?      �?      �?               @      �?              �?      @       @               @      @      �?      @      @      @      @      @      @      @      @      @      @      @      $@      $@      ,@      *@      "@      ,@      8@      8@      0@      8@      =@      >@      :@      >@      A@     �C@      B@     �O@     �G@     �G@      Q@     �N@     �T@     �T@     �X@     �\@     @`@     �_@     �f@      c@     �b@      g@     �g@      l@     0q@     @p@     �s@     0s@     �x@     px@      {@     }@     ��@     Ђ@      �@     ��@     Ј@     ��@     0�@     �@     ��@     <�@     ��@     L�@     �@     ��@     ��@     ��@     �@     ԥ@     Z�@     ��@     D�@     �@     ܰ@     �@     ��@     ��@     Ӹ@     ��@     ,�@    ���@    ���@     ��@    �#�@     #�@    �"�@     ��@    @!�@    @��@    ���@    ���@    ���@    �0�@    @��@    ���@     ��@    �B�@     �@    `-�@    @j�@    ��@    ���@    ���@    @,�@    `��@    ��@     ^�@    �o�@    0��@     *�@    ��@    ��@    ��@    pA�@    ��@    ���@    �%�@    P-�@    �-�@    ���@    pA�@    �t�@    ���@     ��@     n�@    @��@    `�@     J�@    �E�@    ���@    ���@     ��@     �@     ș@     �@      o@      U@      @@      @        
"
CNN3/weights/summaries/mean��_9
&
CNN3/weights/summaries/stddev_1 ��=
!
CNN3/weights/summaries/max��d>
!
CNN3/weights/summaries/min7Ig�
�
 CNN3/weights/summaries/histogram*�	   �&�̿   @q��?      �@! Ti�:�@)��ҷ�p@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[����Zr[v��I��P=��8K�ߝ�a�Ϭ(���(��澢f����5�"�g��>G&�$�>��>M|K�>�_�T�l�>�FF�G ?��[�?����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?�������:�              @      R@     �t@     �@     ��@     ��@     ��@     ؈@     0�@      �@     p�@     �@     ��@     ��@     X�@     H�@     ؂@     ��@     �@      ~@     �z@     �v@     �v@     t@     `s@      o@     �l@     �m@     �f@      h@     �e@     �c@      e@      a@     �Y@     �W@     �V@      U@     @T@      R@      O@      J@      O@      P@     �F@     �G@      C@      D@      <@     �@@      7@      8@      0@      1@      5@      .@      (@      $@      @      &@      $@      "@      $@      @      @       @      @       @      @       @      @       @       @       @      @              �?      �?              @      @      @      �?              �?      �?               @      @      �?      �?      �?               @              �?              �?              �?              �?              �?              �?              @       @      �?              �?      @      @              @      @      @      @      @      @      @      @      @      @      @      @      @       @      @      "@      "@      *@      (@      (@      0@      0@      "@      .@      5@      3@      8@      5@      >@     �B@     �C@      C@      G@     �H@     �E@     �J@      N@     �P@     �R@     �R@     �U@      Z@      X@     @_@     �Z@     �_@     @d@     �c@     @g@      i@     �j@     �o@     �p@     Pr@     0u@     �u@     �x@      {@      }@     �|@     `�@     ��@     Ѓ@     ��@     ��@     ��@     x�@     ��@     x�@     Ȉ@     ��@      �@     @�@     ��@     P@      x@      W@      �?        
!
CNN3/biases/summaries/mean�X�=
%
CNN3/biases/summaries/stddev_1]�;
 
CNN3/biases/summaries/max:��=
 
CNN3/biases/summaries/min���=
�
CNN3/biases/summaries/histogram*�	   @��?   @ǖ�?      P@!  ���@)�]����?20�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:0              @      *@      D@      @        
�
CNN3/activations*�   @�C)@      PA!�`X*��@A)ib))8�RA2�
        �-���q=f^��`{>�����~>5�"�g��>G&�$�>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�������:�
            �CA              �?              �?              �?       @      �?      �?               @              �?              �?              �?       @       @      �?      �?       @      �?      @              @       @      @      @       @      @       @      @      @       @      @       @      $@      *@      $@      *@      ,@      ,@      "@      4@      (@      :@      >@      1@      E@      >@      B@     �B@     �@@      F@     �H@      J@     �P@     �R@     �Q@     �Q@      W@     �X@     @V@     �[@     �_@      `@     �c@     �b@     �g@     �g@     @n@     �m@     @r@      s@     �s@     �w@     �y@     �{@     (�@     ��@     p�@     ؃@     ؃@     �@     ��@     ��@     ��@     Б@     ��@     ��@     �@     H�@     D�@     l�@     ȡ@     r�@     <�@     �@     ��@     ��@     ��@     �@     t�@     o�@     \�@     �@     3�@     ��@    �<�@     �@    ���@     ��@     ��@    �*�@     k�@     C�@    �=�@     ��@    ���@    ���@    @ �@    �7�@     ��@    ��@    ���@    �5�@    ���@     ,�@    @��@    ���@     ��@    ���@    ���@    @�@    �@    0>�@    `/�@    �d�@    �m�@    �)�@    @i�@    `��@    @m�@    ���@    ���@    ���@    @ �@     ��@     ��@     ��@     �@     ��@     	�@     ��@     �@     `�@      |@     �j@     @R@      7@       @        
�$
CNN3/batch_norm*�$	   ��Y�   �J�*@      PA!2	�z��A)>�f�PA2�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾙ѩ�-߾E��a�Wܾ�iD*L�پ��>M|Kվ��~]�[Ӿ�XQ�þ��~��¾�[�=�k��5�"�g���0�6�/n��
�}�����4[_>������m!#���
�%W���*��ڽ>�[�=�k�>;�"�q�>['�?��>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�������:�            ��"A    JG=A    @��@    @��@    �e�@     �@    @�@    ���@     ��@    ���@     ��@     �@     ��@     K�@     ��@    ��@    ���@    ���@     �@     ��@     ��@     �@     ܵ@     	�@     4�@     v�@     ~�@     h�@     T�@     �@     (�@     ��@     ؠ@     ��@     H�@     ��@     �@     ��@     l�@     ��@     ��@     ��@     �@     ��@     ��@     Ђ@     8�@     0~@     `{@      {@      z@     �u@     0u@     p@      o@     @o@     �k@      k@     �e@     �g@     �]@      `@     �\@     �]@     @W@      Y@     @T@     �S@     �R@     �P@      H@      O@     �H@      I@     �C@      8@      :@      ?@      8@      6@      B@      5@      3@      6@      (@      2@      *@      @      @      .@      @      "@      "@      "@       @      @       @       @      @      @      @       @              @               @      �?      @      �?               @              �?      �?      �?      �?              �?      �?              �?              �?      �?              �?              �?              �?              �?               @              �?              �?      �?               @       @              @              @      �?      �?      @      @      @      "@      @      @      @      @      @      @      @      @       @      &@      *@      (@      @      2@      (@      1@      2@      1@      7@     �@@      7@     �@@      :@     �F@      C@     �G@     �J@     �I@     �C@     �P@      Q@      P@     �T@     @U@     �W@     �V@     �_@      `@     �b@     �d@     �f@     `j@     �g@      q@     `p@      q@     Ps@     `v@      x@     {@     0}@      @     �@     ��@     �@     Ȇ@     ��@     P�@     @�@     |�@      �@     ��@     X�@     ��@     P�@     ��@     �@     ��@     Σ@     x�@     �@     8�@     (�@     $�@     ɱ@     �@     ��@     ��@     5�@     �@     ׽@     u�@     ��@    ���@    ���@    ���@    ���@    �A�@    �"�@     ��@    ��@    @��@    �_�@    �F�@    ���@    ���@    @��@    @��@    �Y�@    �K�@    �-�@    ��@     ��@     ��@    @\�@     G�@     ��@    �.�@    �D�@    �#�@    �Z�@     ��@     ��@    @#�@    ���@    @��@     ��@     a�@    ���@     ��@     2�@     �@     ��@     Ԯ@     `�@     �@     ��@     �w@     �d@     �J@      1@      @      �?        
 
accuracy_streaming_mean_1��?|�rG�      �{��	Z�o����A*��
"
CNN1/weights/summaries/meany���
&
CNN1/weights/summaries/stddev_1^�=
!
CNN1/weights/summaries/max`V>
!
CNN1/weights/summaries/min{Z�
�
 CNN1/weights/summaries/histogram*�	   `O@˿   ���?     ��@! @;�F�)��L�&�#@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW��qU���I�
����G��u�w74���82�U�4@@�$��[^:��"���d�r?�5�i}1?�[^:��"?U�4@@�$?uܬ�@8?��%>��:?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @      3@      6@      4@      ;@      ;@      9@      <@      =@      @@      ?@      ?@      <@      <@      =@      7@      6@      4@      1@      ,@      *@      0@      (@      (@      @      &@      "@      @       @      @       @      @       @       @      @      @      @       @      @      @       @      �?       @      �?       @      �?      �?       @      �?      �?               @      �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?       @               @       @       @      @      @      @      @      @       @      @       @      @      @      @      @      @      @      $@      @      &@      @      (@       @      "@      (@      *@      ,@      0@      5@      6@      3@      6@      ?@      ;@      6@      8@      ?@      4@      3@      =@      6@      6@      8@      0@      @        
!
CNN1/biases/summaries/mean[�=
%
CNN1/biases/summaries/stddev_1�<
 
CNN1/biases/summaries/max��>
 
CNN1/biases/summaries/min��=
�
CNN1/biases/summaries/histogram*�	   `7�?    <��?      0@!   �b��?)"	.!ı�?2@� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:@              @       @      @      @      �?       @        
�
CNN1/activations*�    �l@      pA!8ݩ�͇�A)�����A2�
        �-���q=���%�>�uE����>8K�ߝ�>�h���`�>pz�w�7�>I��P=�>��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�*ǣ0?@��#G�'A@C\t��B@J23��D@R��'�F@t�n̫I@��`��K@���cN@Π54�P@�0n9�bR@��X�9T@
6@�F?V@r;`�xX@��iI�Z@~
�i�]@��GL:I`@߱�&�a@C��ôc@�Y��=�e@����*�g@�'�;�:j@}+��
�l@�������:�
           ���dA              �?              �?              �?              �?              �?              �?       @       @              �?      �?      @       @      @       @              @              @      @      @      @      @      @       @      @      @      &@      $@      @      0@      (@      2@       @      $@      *@      2@      6@      2@      3@      :@      3@      ;@     �E@     �A@      B@      A@     �I@      I@     �I@     �O@      L@      V@     @T@     �V@     �Z@      ]@     �]@      ]@     `a@     �d@     `i@     �i@     �m@     �m@     �p@     �s@     �s@     �r@     `w@     �x@     �|@     H�@      �@     ��@     �@     @�@     ��@     �@     ȏ@     `�@     �@     l�@     �@     �@     ��@     �@     ��@     d�@     �@     �@     ԧ@     8�@     �@     v�@     ±@     ��@     f�@     ��@     (�@     �@     ��@    �*�@     ��@    ���@     3�@    �s�@    ��@     #�@     G�@     �@    ���@    @U�@    �d�@    @��@    �A�@    ���@     �@    ���@    �m�@    `w�@    �{�@    @��@    ���@    �C�@     ��@    0��@    ���@    p��@    m�@    �9�@     h A    @�A    �A    �A    P�A    A    ��	A    ��A    ��A    �A    ��A     �A    �0A    A    ȕA    ��A    �A    ��A    0� A    ���@    ��@    @��@    p�@    P$�@    `g�@    0��@     ��@     ��@    �H�@     H�@     ��@     �@     ��@     `n@        
�'
CNN1/batch_norm*�'	   �E�   ���'@      pA!+���b��@)A��S�oA2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þG&�$��5�"�g���0�6�/n���u`P+d����n�����;9��R���5�L�����]���>�5�L�>���?�ګ>����>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�            ($A    \A    �&A    �dVA   ���QA    �1�@    �?�@    p��@    Є�@    ���@    ���@    ���@     ��@    �t�@    @��@    ���@    @��@    `�@    �i�@    ���@    ��@     ��@    ���@    ���@    @��@     ?�@     ��@     �@     p�@      �@     ��@    ���@     ��@     k�@      �@     �@     �@     ��@     ��@     �@     ��@     �@     f�@     �@     ¥@     ��@     ^�@     ��@     T�@     0�@     T�@     p�@     h�@      �@     @�@     ȍ@     8�@     @�@     X�@     ��@     ��@     ��@     @�@      {@     Py@     px@     pv@      s@     `o@     `p@     `p@     �j@     �i@     �d@     `c@     �d@     �^@     �^@     �W@     �^@     @U@     �T@     �U@     �P@     �M@     �O@      K@      H@      H@      B@     �D@      9@      7@      3@      5@      3@      5@      <@      .@      4@      1@      @      (@      (@      @      @       @      *@      @      @      @      @      @       @      @      �?       @       @      @      @      @       @       @               @      �?      �?      �?      �?      �?      �?      �?              �?              �?      �?              �?              �?              �?              �?      �?      �?               @              �?      �?      �?              �?      �?              �?      �?      �?      @       @      @       @       @      @      @       @      @      @      @      @      @      "@      @      &@      ,@      *@      .@      0@      1@      9@      1@      4@      7@      =@      >@      H@     �@@     �G@      L@      L@     �K@      G@     @Q@     �Q@     �T@      S@     �V@     �[@     �`@     �`@     `c@     �d@     �e@     �h@     �i@      l@     `m@      r@     �s@     @t@     x@     0}@     `~@     �~@      �@     @�@     p�@     ��@     ȉ@     ؋@     ȍ@     �@     l�@     ��@     (�@     P�@     ��@     ��@     ��@     |�@      �@     ��@     ��@     *�@     �@     �@     ϱ@     b�@     Q�@     1�@     ��@     �@     \�@    �Y�@    ���@    ���@     ��@     ��@     U�@    ���@    @��@    ��@    @��@     ��@     H�@    @h�@     �@    ���@    �e�@     C�@    ���@    �
�@      �@    `1�@     ��@    �-�@    ���@    �!�@    ��@    p��@    ���@    @��@     k�@    ��@    0� A    ȦA    �A    PA    �9A    �A    � A    `�A    p�A    ��A    %A    @� A    Я�@     ��@    ��@    ���@    �[�@    `r�@    �]�@    @��@    ���@    `W�@    ���@     O�@    ���@     Q�@     ��@     *�@     l�@     x�@     ��@      G@        
"
CNN2/weights/summaries/meanR޺
&
CNN2/weights/summaries/stddev_1�M�=
!
CNN2/weights/summaries/max��_>
!
CNN2/weights/summaries/min��n�
�
 CNN2/weights/summaries/histogram*�	   ���Ϳ   ����?      �@! �2��5�)C0!O�Y@2�
�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !��.����ڋ��vV�R9���d�r�x?�x���_�T�l׾��>M|Kվf�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�
               @     �C@     �e@     �g@     �n@     0p@     Pp@     `q@     �t@     �t@     @r@      t@     �r@     0p@     Pp@      m@      l@      l@     �n@     �f@      d@     ``@     �a@     ``@     @Z@      X@     �[@     �T@     @W@      S@     �N@     �P@     �L@     �G@      J@      A@      F@     �@@      8@      B@      A@      <@      4@      .@      "@      *@      $@      .@      *@      "@      "@       @      (@      $@      @      @      @      �?      @      @      @               @      @      @      @       @      @       @      �?              @      �?              �?      �?              �?               @      �?              �?              �?              �?              �?       @       @              �?      �?      @      �?              �?      �?       @      @      @      �?       @      �?      �?       @      �?      �?      @       @      @      @      @      @      @      @       @      $@      "@      @      "@      (@      &@      "@      1@      0@      (@      $@      5@      9@     �A@      :@     �@@      ?@     �D@      G@      A@      D@      H@     �I@     @P@      Q@     �V@     �P@     �R@     @Y@     �W@     �_@     �^@     �`@     �c@     �d@     �e@      h@     �i@      l@      l@     �o@     �o@     �q@     �r@     �q@     �p@     pr@     �u@     �p@     @r@     �j@     �e@      a@      7@        
!
CNN2/biases/summaries/mean�6�=
%
CNN2/biases/summaries/stddev_1���;
 
CNN2/biases/summaries/max��=
 
CNN2/biases/summaries/min��=
�
CNN2/biases/summaries/histogram*�	   `��?   ��|�?      @@!   ئ	@)r������?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      2@      &@        
�
CNN2/activations*�   ���*@      `A!Ȓ�y�V_A)!�ZnEsA2�
        �-���q=�H5�8�t>�i����v>�5�L�>;9��R�>豪}0ڰ>��n����>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�������:�
           �Z�FA              �?              �?              �?              �?               @              �?      �?       @      �?       @      @      �?      @      @              @       @       @      @       @       @      @              @      @      @      @      "@      ,@      &@      *@      $@      ,@      "@      .@      .@      (@      9@      A@      *@      :@      ;@      B@      @@     �A@      H@      I@     �G@     �M@      O@      M@     �Q@     �S@     �U@      Y@     �Y@      ]@      _@      b@     @c@     �e@     �f@     �h@     �h@     `p@      r@     �q@     �s@     �v@     �y@     �y@     �|@      �@     ��@     �@     �@     ��@     ��@     H�@     ��@     ܑ@     Ԓ@     ؔ@     (�@     0�@     �@     0�@     ֠@     ��@     T�@     .�@     �@     ج@     ��@     ��@     s�@     ��@     u�@     ݸ@     ��@     �@    �{�@     _�@    �v�@     y�@     6�@     m�@     �@    ���@    �f�@    @f�@    @f�@    ���@    @a�@    �B�@    ���@     ��@    ���@    �U�@    ��@    `��@    p@�@    �?�@    �^�@    @��@    ���@    �>�@    `j A    h�A    h�A    8�A    �A    �u
A    `"A    �/A    H�A    �@A    SA     "A     pA    8�
A    0�A    �A    �A    ��A    �A    P~�@    ��@     �@    ���@    ���@     &�@     ��@     �@     ��@     Ȏ@     pq@     �Y@      2@      @       @        
�'
CNN2/batch_norm*�'	    ���   `:o%@      `A!p�����@)gt� A `A2��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿ['�?�;;�"�qʾ
�/eq
Ⱦ����ž0�6�/n���u`P+d����n�����豪}0ڰ��MZ��K���u��gr��R%������
�}�����4[_>�����z!�?�>��ӤP��>豪}0ڰ>��n����>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@�������:�            ��(A    9>A     �A    `�A    { A    pA�@    �,�@    p��@    p��@    ���@    �!�@    �Y�@    ���@    p;�@    `��@    ���@    @:�@    @��@    ��@    ���@     ��@    `3�@     ��@    ���@     6�@    ���@    �#�@    @�@    �+�@    �[�@     ��@    ���@     G�@     #�@     F�@    �1�@     ��@     V�@     �@     ��@     ʹ@     �@     �@     ̮@     (�@     �@     �@     �@     ��@     8�@     |�@     ��@     �@     ��@     4�@     �@     x�@     ,�@     ��@     H�@     ��@     ��@     ��@     p�@     ��@     �|@     Pz@     �w@      w@     �t@     �q@     Pq@     `q@     `l@     �j@     �e@     @c@     �b@     �_@      `@     �]@     @Y@     @\@      W@     �V@      P@     �S@      K@      O@      K@     �E@      =@      F@      @@      ?@      9@      8@      B@      6@      6@      2@      2@      ,@       @      &@      @      &@      @      $@      "@      @      @      @      @      @       @      @       @      @      @      @      �?      �?      �?      �?       @      �?       @              �?      �?      �?              �?              �?              �?      �?              �?              �?              �?               @      �?      �?              �?      �?      �?              �?       @       @      �?      �?               @      �?      �?      @      @      @      �?       @      @      @      @      @      @      @      "@      "@      $@      *@       @      ,@      *@      $@      .@      7@      2@      7@      7@      ;@      =@      >@      C@      C@      >@      G@     �F@      H@     �L@      M@     �R@      T@     �U@     @T@     �V@     �X@     �_@      b@      d@     �f@     �h@     �i@     �i@     �o@     �q@     �r@      t@     pu@     �z@     `{@     �}@     ��@     p�@     �@      �@     @�@     �@     x�@     t�@     `�@     ��@     P�@     @�@     @�@     @�@      �@     8�@     8�@     `�@     6�@     ��@     �@     ��@     �@     S�@     ³@     g�@     ��@     �@     ;�@     �@    ���@    �E�@     I�@    �K�@     ��@    �X�@     ��@    @��@    ���@    �A�@    �)�@    �7�@    @t�@    @M�@    @��@    @r�@    @�@     ��@    �V�@    `�@     "�@     S�@    ���@    p��@    p��@    �`�@    ��@    @��@    .�@    ��@    ���@    @��@    `}�@    ���@    �K�@    `��@    �3�@    ���@    �d�@    ���@    (�@    0��@    ���@    p��@    �s�@    `��@    `o�@    ��@     �@    �o�@     ��@     \�@     $�@     ��@     4�@     h�@     �r@     @_@     �B@       @       @        
"
CNN3/weights/summaries/mean#��9
&
CNN3/weights/summaries/stddev_1���=
!
CNN3/weights/summaries/maxf1f>
!
CNN3/weights/summaries/min��l�
�
 CNN3/weights/summaries/histogram*�	   ���Ϳ   �,��?      �@! �(�1 @)���<p@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`��%ᾙѩ�-߾
�/eq
Ⱦ����žjqs&\��>��~]�[�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?�������:�              @     �R@     �t@     p�@     ��@     ��@     x�@     8�@     ��@     X�@     ��@     ��@     x�@     ��@     ��@     p�@     8�@     x�@     Ȁ@     �}@     �y@      x@     �v@     �t@     �q@     �p@     `m@     �k@      g@      h@     @g@     �a@     �b@     @`@     �[@     �[@     �W@     @W@     @S@     @Q@     �Q@      P@      E@      P@      K@      E@     �B@     �B@     �D@      <@      :@      4@      4@      3@      3@      @      &@      "@      &@      "@      "@      @      "@      @      "@      @               @       @       @      @      @      �?      @       @       @      �?       @      @      �?       @      @       @       @      @              �?      �?       @               @      �?              �?      �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?               @              @              �?               @       @      �?       @      @      @      @      �?      @      @      @      @      @      @      &@       @      $@      @      ,@      *@      "@      .@      6@      (@      4@      (@      <@      4@      9@      @@      >@     �B@     �A@     �D@     �H@      J@      L@      M@     �M@     �T@     �T@     �T@      R@     @_@      ]@     �\@     �a@     `b@      c@     �e@     @h@     `n@     @n@     r@     `q@     �t@     �u@     �y@     @z@     @|@     �~@     ��@     x�@     �@     ��@     ��@     ��@     ��@     ȉ@     ��@     X�@     p�@     ��@     �@     x�@     �@     �w@      W@       @        
!
CNN3/biases/summaries/mean���=
%
CNN3/biases/summaries/stddev_1�է;
 
CNN3/biases/summaries/max�p�=
 
CNN3/biases/summaries/min���=
�
CNN3/biases/summaries/histogram*�	   @���?   @.�?      P@!   �x@)P����?20�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:0               @      .@     �F@       @        
�
CNN3/activations*�   `�.+@      PA!pB���?A)<��m�\RA2�
        �-���q=.��fc��>39W$:��>���]���>�5�L�>��~���>�XQ��>['�?��>K+�E���>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�������:�
           �<eCA              �?              �?              �?               @              �?       @      �?       @      @      �?      �?      @      �?               @              @              @      @      @      @      @      @      @      @      �?      @      @       @      &@      .@      *@      "@      ,@      .@      0@      5@      6@      2@      4@      ;@      A@      ?@      G@     �F@      D@     �H@      G@     �O@      M@      O@     �S@     @T@      R@      Y@     �Z@     `b@     �_@     �^@      f@      e@      k@      k@     �k@     �m@     pp@     �r@     `v@     �z@      v@     �|@     �~@     ��@     x�@     (�@     x�@     (�@     ��@     ��@     ��@     ��@     ��@     D�@     ��@     ��@     @�@     �@     ġ@     ң@     $�@     ��@     ��@     t�@     �@     h�@     O�@     ״@     ��@     θ@     ��@     W�@    �i�@    �y�@     �@    ���@    ��@    ���@    ���@     ��@    �J�@    ���@    @��@     ��@    @y�@     ��@     R�@     ��@    @]�@     ��@    ��@    ���@    �r�@    @�@     ��@    �?�@     ��@     ��@    `��@    �8�@    �L�@    ��@    �d�@    `��@    ��@    @=�@    @��@    `R�@     ��@    ���@     p�@    @��@     �@    �&�@    ���@     ��@     ��@     گ@     z�@     ̙@     �@     @m@     �X@      6@      @      �?        
�#
CNN3/batch_norm*�#	    7�   ���*@      PA!�"�nA)p�:ՌPA2�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾙ѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾5�"�g���0�6�/n���u`P+d����n�����['�?��>K+�E���>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�������:�            �A    y�:A    ��$A     ,�@     d�@    ���@    ���@    �!�@    ���@    ���@     ��@    ���@     �@    �I�@     ��@     ��@    �s�@     ��@    ��@     ��@     �@     7�@     \�@     ��@     ñ@     p�@     ��@     �@     ��@     $�@     �@     H�@     �@     �@     0�@     x�@     $�@      �@     (�@     ��@     (�@     @�@     �@     ȇ@     H�@     X�@      �@     �~@     �{@     �x@     �w@     u@     �r@     0q@     �p@     @p@     `h@     �g@     �f@     �e@     �b@     @_@     �X@     �_@     �[@     �W@      U@     �S@     �Q@     @P@      L@      J@      B@      H@      N@      >@      ?@      ;@      1@      =@      :@      :@      0@      *@      "@      "@      &@      "@      ,@       @      "@      @      0@      @      "@      @      @      @      @      �?      @              @       @              �?              �?       @      �?       @      �?              �?               @      �?      �?              �?              �?              �?              �?              �?      �?      @              �?       @       @      �?               @      @      �?      @      @      @      �?      @      @      @      @      @      @      @      @      @      @      @      "@      4@      &@      *@      2@      ,@      2@      ,@      5@      ;@      5@      9@     �@@      ?@      =@      B@     �B@      I@      N@     �L@     �K@     �Q@     @R@     @T@     �U@     �U@     �W@      Z@     �_@     @`@      d@      d@      k@     �i@      m@     �q@     @r@     �s@     `u@     �v@     0|@     0{@     `@     ��@     ��@      �@     ��@     0�@     X�@     ��@     8�@     ԑ@     ��@     ��@     ,�@     �@     ��@     П@     ��@     d�@     \�@     P�@     ��@     ��@     V�@     �@     ��@     @�@     '�@     >�@     ��@     ��@     g�@     �@     ��@    �9�@     e�@    �w�@    ���@     y�@     b�@     �@     ��@     �@    ���@     ��@    ���@    @��@    ���@    �]�@    ���@    @c�@    `��@    �T�@    ���@    ���@    ���@     ��@    ���@     d�@    @�@    �U�@     V�@    ���@     ��@    @0�@    ���@     ��@    @`�@     	�@    ���@     ��@     �@     ��@     ��@     f�@     4�@     ��@     0�@     �h@     �V@      <@      @       @        
 
accuracy_streaming_mean_1��?q&%y��      �[1a	�/@���A*��
"
CNN1/weights/summaries/meanRW��
&
CNN1/weights/summaries/stddev_1���=
!
CNN1/weights/summaries/max�mW>
!
CNN1/weights/summaries/min��U�
�
 CNN1/weights/summaries/histogram*�	    8�ʿ    ���?     ��@!  ~jʺ�)g-�ت�#@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed��l�P�`�E��{��^���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L����#@�d�\D�X=���82���bȬ�0�ji6�9�?�S�F !?�[^:��"?U�4@@�$?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?�T���C?a�$��{E?
����G?��bB�SY?�m9�H�[?E��{��^?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @      3@      3@      6@      8@      =@      ;@      <@      <@     �@@      >@      @@      <@      =@      9@      5@      :@      4@      ,@      ,@      .@      3@      $@      (@      @      &@      $@      @      @       @      @      �?      @      @       @      @      @      @      @      @      �?      �?              �?               @       @               @              �?              �?      �?      �?      �?               @              �?              �?              �?              �?      �?              �?              �?      �?              �?      �?               @              �?       @      �?      �?       @      �?      @      @      @      @      �?      @      @      @      @      @      @      @      "@      @      (@      @      $@       @      (@      &@      ,@      *@      (@      2@      ;@      5@      6@      ;@      <@      6@      :@      <@      6@      4@      =@      5@      6@      7@      1@      @        
!
CNN1/biases/summaries/meanD��=
%
CNN1/biases/summaries/stddev_1Bc�<
 
CNN1/biases/summaries/max�{>
 
CNN1/biases/summaries/minq��=
�
CNN1/biases/summaries/histogram*�	    ��?   �s/�?      0@!   V��?)�;1�=�?2@� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:@              @      @       @      @      �?       @        
�
CNN1/activations*�   ���k@      pA!�`#���A)V���7��A2�
        �-���q=�f����>��(���>pz�w�7�>I��P=�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�*ǣ0?@��#G�'A@C\t��B@J23��D@R��'�F@t�n̫I@��`��K@���cN@Π54�P@�0n9�bR@��X�9T@
6@�F?V@r;`�xX@��iI�Z@~
�i�]@��GL:I`@߱�&�a@C��ôc@�Y��=�e@����*�g@�'�;�:j@}+��
�l@�������:�
           �P�dA              �?              �?              �?              �?              �?       @              �?               @       @       @              @       @      �?      @       @       @      @      �?       @      @      @       @       @      @      @      "@       @      *@      (@      &@      ,@      *@      *@      0@      7@      2@      2@      0@      <@      ;@      B@     �C@      B@     �@@      M@      L@     �P@      J@     �S@      O@      U@      W@      Y@     @\@     �[@      a@     `d@     �f@     @f@     @e@     �i@     @m@      n@      p@     0s@     �v@     x@     �z@     `x@      ~@     @�@     8�@     ��@     @�@     �@     �@     ��@     ��@     Б@     ��@     ��@     �@     Ě@     ��@     b�@     B�@     B�@     "�@     R�@     ک@     :�@     Z�@     d�@     ��@     Դ@     ��@     ݸ@     6�@     C�@    �z�@     ��@     ��@    ��@    ���@     ��@     ��@    � �@     ��@    @��@    �)�@    @��@    ��@    �~�@     ��@    �*�@    ���@    ���@    �0�@    ��@    @i�@    @��@    �y�@    `�@    P��@     2�@    P��@    �X�@    ���@     ��@    �' A    �CA    XvA     �A    ��A     �	A    A    @�A    �KA    hvA    ��
A    �[A    �OA    x
A    �D
A    �A    ��A    �$A    �A    �WA    p��@    p��@     ��@    �H�@    P:�@    �?�@    @��@    @��@    �c�@     ��@     >�@     ��@     8�@      `@        
�'
CNN1/batch_norm*�'	    �~�   ��u'@      pA!Q{*��U�@)�{ϓH�oA2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ�XQ�þ��~��¾�[�=�k��豪}0ڰ���������?�ګ�;9��R�����]������|�~�������0c�w&���qa��MZ��K�>��|�~�>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�            x2A    �IA    `NA   �"-WA   ���PA    P>�@    @��@    ���@     ��@    ���@    �6�@     "�@    @g�@     =�@     �@    `��@     �@    �d�@    ���@    �F�@    @��@     ��@     ��@    @u�@    @��@     ��@     ��@     ��@    ���@     ��@    �t�@    �~�@     Y�@     �@     �@     ��@     µ@     �@     ڱ@     �@     ��@     ��@     |�@     ��@     .�@     @�@     V�@     ��@     �@     P�@     4�@     ȕ@     �@     ��@      �@     ��@     ��@     X�@     ��@     ��@     P�@     8�@     ~@      {@     �w@     �v@      r@     �r@     q@      m@     `i@      g@     @i@     �e@      d@     �`@     �^@     �Y@     �X@      V@     @U@     @P@      S@      P@      M@      I@      H@      B@     �B@     �B@      9@      ?@      9@      <@      6@      *@      *@      1@      8@      0@      (@      "@      ,@      $@      "@      @      $@      @      @      @       @       @      @       @      @      @               @      �?              �?      �?      �?               @              @      �?       @      �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?      �?              �?      �?      �?      �?      �?       @       @              @       @      @      @       @      @      �?      @      "@       @      @      �?       @      @      @      ,@      @      $@      3@      7@      1@      "@      2@      6@      ,@      8@      .@      @@      2@      ?@      ;@      >@      I@      =@     �L@      G@     �M@      N@     �V@      T@     �R@     @S@     @\@     �`@     �`@     `b@     �b@     �c@     �e@     @i@      k@      o@      q@      t@      s@     0u@     @y@     `x@     @{@     p�@     ��@      �@     0�@     P�@     ��@     �@     ��@     $�@     �@     <�@      �@     Ԙ@     $�@     �@     Ԡ@     �@     <�@     �@     V�@     ��@     ��@     ��@     `�@     ��@     ��@     �@     K�@     ��@     ��@    ���@     ��@    ���@     ��@     ��@    ��@    ��@    ���@     ��@     ��@    @��@    �c�@    �o�@     2�@    `��@     ��@    �7�@     k�@    ���@    ���@    `%�@    P��@    �=�@    p��@    ���@    ���@    `U�@     	�@    ���@    P��@    ���@    H4 A     � A    �RA     �A    8�A    p@A    HA    �?A    8�A    �uA    mA    P��@    ���@    ���@    � A    p��@    P��@    ���@     �@    ���@    ���@    ���@    �^�@     ��@    ���@     4�@     N�@     ܗ@     ��@     ��@      Q@        
"
CNN2/weights/summaries/mean!���
&
CNN2/weights/summaries/stddev_17u�=
!
CNN2/weights/summaries/max�a`>
!
CNN2/weights/summaries/min"o�
�
 CNN2/weights/summaries/histogram*�	   @�Ϳ   `8�?      �@! 82�28�)��D��!Y@2�
�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�ji6�9���.����ڋ��T7����5�i}1���d�r�['�?�;;�"�qʾ��|�~�>���]���>��d�r?�5�i}1?�T7��?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�
               @      D@     �e@      h@     @o@     �p@     pp@     0q@     �t@     �t@     �r@     `t@     �q@     �p@     �p@     �m@     �k@      o@     �i@     �f@     @e@     �`@      a@     @_@     @[@     �]@     @W@     �U@     �S@     �T@     �O@      P@      L@     �M@      I@     �D@      >@      =@     �B@      3@      >@      ?@      3@      0@      6@      &@      *@      &@      &@      (@      &@      "@      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      �?      @      �?      �?              �?      �?              �?       @              �?      �?              �?              �?              �?      �?              @               @               @               @       @      @      �?      �?      �?              �?      @       @      @      @      @      @      "@      @      $@      $@      $@      &@      @      &@      .@      *@      &@      7@      <@      7@      3@      =@      8@      E@     �D@     �F@     �F@      B@     �M@      G@      M@     �Q@     �Q@      W@      R@     �T@     @Z@     @_@     ``@     `a@     �b@      d@      e@     `g@      i@     @m@     @n@     �l@     q@     q@     s@     `r@     0q@     �q@      u@     0p@      r@      k@     �f@     @`@      =@        
!
CNN2/biases/summaries/mean,!�=
%
CNN2/biases/summaries/stddev_1�]�;
 
CNN2/biases/summaries/max&:�=
 
CNN2/biases/summaries/min�ػ=
�
CNN2/biases/summaries/histogram*�	    {�?   �DG�?      @@!   �%�	@)�FS獗�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              @      2@      $@        
�
CNN2/activations*�   � �(@      `A!�`����^A)��'��sA2�
        �-���q=f^��`{>�����~>[#=�؏�>K���7�>;9��R�>���?�ګ>豪}0ڰ>��n����>0�6�/n�>5�"�g��>��~���>�XQ��>�����>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�������:�
            ^�GA              �?              �?              �?              �?              �?              �?      �?              �?              �?               @      �?              �?       @      @               @       @              @      �?      @       @      @       @      @      @      @      @      @      $@      $@      @      (@      0@      *@      ,@      *@      *@      $@      ,@      0@      3@      1@      >@     �G@      >@      @@      H@     �F@     �G@     �M@     �L@     �P@     @P@     @Q@     @V@     �V@     �X@     �Y@      ]@      c@      `@     �e@      g@     �j@      g@     �l@      q@     �p@     �s@     �t@     �u@     �|@     �}@     @@     ��@     ��@     ��@     �@     ��@     �@     `�@     �@     |�@     ��@     |�@     ��@     ��@     ��@     �@     ��@     �@     ئ@     �@     �@     ԭ@     ��@     L�@     �@     s�@     �@     .�@     �@    �3�@    ��@     ;�@     ��@    �h�@     ��@    ���@    �T�@     &�@    �-�@     N�@     ��@    �E�@    @c�@    `��@    ��@    ���@    ��@    ���@    @r�@    `��@    ��@    ���@    ��@    @f�@     b�@    U�@    (� A    ��A    A     }A      
A    x�A    ��A    @�A    �XA    �%A    H�A    ��
A    -	A    �A     ;A    0�A    hA    X^A    P��@     g�@    �}�@     ��@    ���@    ���@     �@     մ@     ^�@     �@     �v@     @_@      =@      @        
�(
CNN2/batch_norm*�(	   �C)�   `@�%@      `A!�j���@)p?�� `A2��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g����u`P+d����n�����豪}0ڰ�������u��gr��R%������39W$:���.��fc���u��6
�>T�L<�>�5�L�>;9��R�>���?�ګ>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@�������:�            �A   �#JBA    hA    LPA    0 A    �<�@    �#�@     �@    P�@     /�@    �Q�@    ��@    �1�@    ���@    `�@    @��@     $�@     W�@    @`�@     ��@    `�@    ���@     J�@    @��@    @o�@    ��@    @��@    ���@    ��@     E�@    ���@     �@     ��@    ���@    ��@     ��@     ��@     �@     ��@     0�@     �@     U�@     հ@     r�@     2�@     �@     ��@     ��@     �@     N�@     D�@     �@     ��@     @�@     ��@     ��@     ؑ@     Ȑ@      �@     ��@     ��@     ��@     �@     ��@     ��@     �}@     `~@     0{@     �t@     �s@     �t@     �p@     �l@      l@     �h@     �f@     �a@      b@     `c@      `@     @_@     �[@     �V@     �T@     �U@     �O@     �Q@      Q@     �G@      F@      G@      B@      C@      9@      F@      <@      4@      <@      6@      4@      *@      1@       @      ,@      0@      (@      &@      *@      "@      $@      $@      @      @      @      @      @       @      @      �?      @      @      �?       @      @      �?              �?               @      @              �?              �?      �?              �?       @      �?              �?              �?              �?              �?              �?              �?      �?              �?       @       @               @              �?      �?      @       @       @      �?       @       @       @      @       @      @      @      �?      �?      �?      @      �?      @      @      @      @      @       @      @      "@      @      *@      &@      @      0@      "@      1@      ,@      9@      4@      <@      =@     �A@      >@      ;@      A@      :@     �H@      >@      P@     �F@     �O@     �U@     �N@     �S@      T@      X@     @Z@      ^@     @_@      b@      c@     `e@     �e@     `h@     �l@     �o@     q@     0s@      t@     �w@     0y@     `|@     ��@     0�@     x�@     �@     `�@     X�@     �@     ؎@     ,�@     ��@     �@     8�@     ��@     �@     �@     ԝ@     ,�@     �@     ��@     �@     �@     �@     ��@     -�@     ǲ@     8�@     ��@     ��@     l�@     ܽ@    �|�@    ���@     X�@     V�@     j�@     ��@     ��@     R�@    ���@    @v�@    @d�@    ���@    @��@    �[�@    ��@    `��@    ���@     M�@    �I�@    `4�@     ;�@    �q�@    ���@    ���@    ���@    ��@    ���@    ���@    @b�@    p�@    ���@    0�@    �,�@    ���@    @n�@    �@    `�@    @6�@    ��@    ��@      �@    �|�@    ��@    �b�@    ���@    � �@     ��@    ���@    ���@     z�@    �v�@    ��@     �@     �@     ڮ@     `�@     ��@     0t@     @Z@      B@      &@      @        
"
CNN3/weights/summaries/mean�ٚ9
&
CNN3/weights/summaries/stddev_1Sߵ=
!
CNN3/weights/summaries/maxqg>
!
CNN3/weights/summaries/min��m�
�
 CNN3/weights/summaries/histogram*�	    u�Ϳ    .��?      �@!���4[#@)�OLo�&p@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�������ߊ4F��h���`�8K�ߝ��uE���⾮��%��u��gr��R%��������ӤP��>�
�%W�>��n����>�u`P+d�>a�Ϭ(�>8K�ߝ�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?�������:�               @     �Q@     �u@      �@     ��@     �@     ��@     X�@     P�@     ��@     ؆@     0�@     Ѕ@     ��@     ��@     ��@     ��@     p�@     ��@      ~@     �y@     @x@     Pt@     �u@     �q@     �p@     �o@     `k@      h@      h@      f@     �a@     �b@     ``@     @Z@      \@     �V@     �V@     �V@      T@     �P@      P@      K@      J@      C@     �A@     �E@      D@     �A@      @@      8@      9@      5@      1@      ,@      2@      *@      *@       @      $@      $@      (@      $@      @      @       @      @      @       @      @      @       @      @      @      �?       @              @               @      �?       @       @              �?      �?      �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?               @               @      �?              @       @       @      �?      �?       @       @      �?       @       @      @      @      @      @       @      @       @      $@       @      *@       @      @      0@      @      .@      .@      2@      4@      5@      @@      4@     �A@      A@     �F@      J@      ?@     �G@     �F@     @Q@     @P@     @S@      X@     �V@      U@     �X@     �^@     @]@     �b@     @^@     �f@     �f@      i@     �k@     �n@     �r@     �q@     �r@     �u@     `y@     �{@     �{@     `~@     �@     ��@     ��@     �@     @�@     Ѕ@     ��@     ��@     ��@     h�@     X�@     P�@      �@     h�@     �@     �v@     �Z@       @        
!
CNN3/biases/summaries/mean��=
%
CNN3/biases/summaries/stddev_1�"�;
 
CNN3/biases/summaries/max��=
 
CNN3/biases/summaries/min��=
�
CNN3/biases/summaries/histogram*�	   �x"�?   �?`�?      P@!   9�`@)b�#���?20�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:0              @      ,@      E@      @        
�
CNN3/activations*�   `��+@      PA! �'r�>A)3�f��(RA2�
        �-���q=��|�~�>���]���>;9��R�>���?�ګ>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�������:�
           ���CA              �?              �?              �?               @      @       @       @              �?              �?      @      �?      @      @              @       @      @       @      "@      @      @      @      @      @      @      (@      *@       @      @      (@      .@      8@      &@      8@      &@      *@      ?@      ;@      @@      A@      >@      A@     �A@     �H@      H@      P@     @R@     @T@     �T@      T@     �Z@     �W@      ]@     �`@     �^@      c@      d@     `g@      k@     �k@     �k@     �m@     @q@      u@     Pu@     �z@     �{@     �{@     �~@     Ђ@     ��@     �@     p�@     ��@     ��@     ��@     ؐ@     ��@     ��@     p�@     D�@     l�@     $�@     �@     
�@     �@     B�@     (�@     h�@     �@     J�@     	�@     �@     ��@     o�@     ��@     x�@     �@    � �@    �%�@    ���@    ���@    �,�@    �M�@     ��@    @��@     J�@    ���@     ��@    @{�@    ���@    ���@    @G�@    �M�@    ���@    ��@     u�@    `�@    `��@    `�@     �@     d�@    @;�@    `��@     `�@    ���@     ��@    ��@    `�@    ��@    ���@    �}�@    ���@     :�@    @��@    @��@     O�@    �Q�@     x�@    �p�@     +�@     ̸@     1�@     <�@     h�@     ؇@     `i@     @P@      3@       @      @        
�$
CNN3/batch_norm*�$	   �b��   `��-@      PA!�Dd��A)-%��PA2�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(��uE���⾮��%ᾙѩ�-߾��~]�[Ӿjqs&\�Ѿ['�?�;;�"�qʾ�*��ڽ�G&�$����|�~���MZ��K���4[_>������m!#�������~>[#=�؏�>�u`P+d�>0�6�/n�>;�"�q�>['�?��>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�������:�            hA    �*A    �S>A    �_�@    @o�@    @
�@    �u�@    ���@    ���@    ���@     �@     .�@    ���@    �,�@    �x�@    ���@    ���@    ���@     ҽ@     /�@     ��@     ׶@     ˴@     �@     ��@     έ@     (�@     N�@     J�@     B�@     ��@     :�@     8�@     `�@     D�@     $�@     h�@     (�@     ��@     ��@     `�@      �@      �@     P�@     P�@     ��@     p@     P}@     |@      x@     �u@      w@     �s@     `o@     �o@     �f@     �h@     @h@     �g@     �`@     �]@     �^@     @[@     �[@      W@     @Q@      R@     �R@     �Q@     �M@     �P@      @@      A@      E@      G@      9@      <@      :@      4@      <@      3@      .@      $@      3@      ,@       @      &@      0@      "@       @      @       @      @      $@      "@      @      @       @      @      @      �?      @      �?       @       @      �?              �?               @               @      �?              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?       @       @       @      �?      @       @      �?      @      @      �?      �?      @      @       @      @       @      @      $@      @       @       @      $@      ,@       @      "@      $@      *@      ,@      7@      0@      2@      ,@      =@      >@     �A@     �H@      H@     �D@      I@      I@     �G@     @P@      O@     �Q@     �S@     �W@     �[@     �X@     @Z@      a@     �]@      a@     �g@     �g@     �h@     �h@     �m@     @p@     �r@      r@     @t@     �y@     �z@     �{@     ��@     ��@     ��@     �@      �@     P�@     Ȍ@     `�@     �@     �@     (�@     ��@     ��@     ��@     ��@     ��@     �@     (�@     ��@     J�@     Ъ@     d�@     ��@     P�@      �@     ��@     -�@     �@     L�@     w�@    ���@     v�@    �.�@     ��@    �5�@    ���@    ���@    ��@    �)�@    @d�@     ��@    ���@    @�@    ���@    ���@     �@    �F�@     `�@    @��@    ���@    ���@     ��@    ���@     ��@    ���@     ��@     �@    @�@     ��@    @��@    �(�@     l�@    �=�@    ��@    @��@     ��@    ��@     [�@     ��@    �q�@     ��@     ��@     <�@     X�@     D�@     X�@     �f@     �Q@      ,@      @      �?      @        
 
accuracy_streaming_mean_1��?5�VY��      ����	���̙��A	*��
"
CNN1/weights/summaries/mean'��
&
CNN1/weights/summaries/stddev_1��=
!
CNN1/weights/summaries/max�Z>
!
CNN1/weights/summaries/min^V�
�
 CNN1/weights/summaries/histogram*�	    ��ʿ   @vA�?     ��@! \�9���)f��#@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�a�$��{E��T���C���%>��:�uܬ�@8�E��a�Wܾ�iD*L�پ�T7��?�vV�R9?U�4@@�$?+A�F�&?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?�lDZrS?<DKc��T?ܗ�SsW?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @      3@      5@      5@      7@      ?@      :@      ;@      =@     �@@      >@      >@      ?@      9@      9@      9@      ?@      .@      (@      ,@      4@      0@      &@       @      "@      $@      "@      @      $@      @      @      @      @      @      @      @      @       @      @      @              @              �?              �?              �?      �?      �?               @      @      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              �?      �?              @       @      @              �?               @      @      @      @      @       @      @      @      @      @      @      @      @      "@      @      (@       @      "@      $@       @      ,@      ,@      *@      *@      *@      =@      5@      7@      ;@      ;@      4@      >@      <@      3@      6@      =@      4@      8@      7@      1@      @        
!
CNN1/biases/summaries/mean��=
%
CNN1/biases/summaries/stddev_1���<
 
CNN1/biases/summaries/max|�>
 
CNN1/biases/summaries/min�y�=
�
CNN1/biases/summaries/histogram*�	   `<��?   ���?      0@!   ��q�?)h?�V"��?2@� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:@              @      @      �?      @      �?       @        
�
CNN1/activations*�    o�k@      pA!xj2ᖤ�A)�	����A2�
        �-���q=�h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�*ǣ0?@��#G�'A@C\t��B@J23��D@R��'�F@t�n̫I@��`��K@���cN@Π54�P@�0n9�bR@��X�9T@
6@�F?V@r;`�xX@��iI�Z@~
�i�]@��GL:I`@߱�&�a@C��ôc@�Y��=�e@����*�g@�'�;�:j@}+��
�l@�������:�
           `�dA              �?              �?       @       @               @               @              �?      �?               @      �?       @      @       @      @      @      @       @       @      @      @       @      @      @      @       @      @       @      .@      "@       @      &@      7@      .@      7@      >@      6@      6@      4@      6@      ?@     �@@     �F@      ?@     �D@     �J@      M@      O@     @T@      P@     @T@     @P@     @\@     @V@     @^@     �`@     �\@     `f@     `c@      g@     `j@      l@     @n@     �p@     s@     �t@     0u@     �w@     0|@     `}@     x�@     ��@     �@     ��@     ��@      �@      �@     p�@     Б@     p�@     ��@     h�@     �@     ��@     �@     Р@     h�@     x�@     �@     ҩ@     �@     ƭ@     ��@      �@     �@     ]�@     D�@     ��@     �@     ־@    �s�@     �@    �<�@    ���@    �J�@    ���@     ��@     ��@     ��@    @�@    ���@     ��@    ���@    �)�@     8�@    ���@    ���@     *�@    ���@     �@    �[�@    `��@     K�@    �<�@    ��@    pn�@    ���@    `'�@    �?�@    ���@     8�@    `JA    ȡA    @�A    �rA    ��	A    �A    �A    p
A    ��A    `rA    �tA    �VA    �}A    �!A    �A    x�A    #A     A    `z�@    ��@    �d�@    p5�@    `A�@     ��@    ���@     �@    �(�@     0�@     &�@     �@      }@       @        
�(
CNN1/batch_norm*�(	    �t�    �|'@      pA!c����@)LW�. pA2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ����ž�XQ�þ��~��¾�[�=�k��G&�$��5�"�g���0�6�/n����������?�ګ�.��fc���X$�z��w&���qa>�����0c>�����~>[#=�؏�>T�L<�>��z!�?�>X$�z�>.��fc��>;9��R�>���?�ګ>����>豪}0ڰ>0�6�/n�>5�"�g��>�*��ڽ>�[�=�k�>
�/eq
�>;�"�q�>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�            � A    VA   @^PA    �WXA    �J�@    ���@     �@    �n�@    ���@    ���@    �V�@    �`�@     ��@    �K�@    �>�@     D�@    ���@     ��@     ��@     ��@    @��@    @��@    �y�@    ��@     A�@     [�@    ���@     ��@     ��@     �@    ���@     ��@     ^�@     9�@     ��@     �@     �@     ��@     )�@     ��@     ��@     ��@     ��@     ��@     r�@     Ơ@     p�@     Ț@     ��@     �@     ��@     T�@     �@     X�@     ��@     ��@     ��@     8�@     ��@     ȁ@     �@     `~@     �{@     pw@     �u@     ps@     �m@     pp@     @l@     `j@     `h@     @g@     `e@     �b@     �c@      Z@     �\@     @[@     �U@     �Q@     �S@      S@      N@      N@     �L@     �C@     �B@     �B@      A@      @@      8@      8@      8@      &@      4@      2@      0@      ,@       @      @      $@      .@      @       @       @      @      "@      @      @      �?      @      �?      @      @      @      @      @      �?       @      �?      �?      @      @               @       @              �?      @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @              �?              �?              �?      �?      �?      @              @       @      �?       @              @       @      @      @      @      @      @      @      @      @      @      @       @      "@      $@      $@      &@      (@      4@      .@      *@      4@      .@      <@      0@      2@      =@      >@     �A@     �E@     �E@      P@      L@     @P@     @S@     �L@     �J@     @X@     @X@     �U@     @[@     @\@     �_@      d@     �d@     �d@     �i@      n@     �o@     0q@     �p@     �t@     �u@     �v@      x@     0|@     P~@     X�@     ��@     p�@     ��@     ؉@     ��@     8�@     ��@     Ԓ@     ܔ@     ��@     �@     �@     \�@     �@     �@     d�@     �@     $�@     ��@     Z�@     D�@     ��@     �@     |�@     $�@     ��@     ӻ@     ��@    ���@    ���@    �w�@     ��@    ���@    ��@    ���@    �q�@    ���@    ���@     ��@    �3�@     ��@    @Q�@    ��@    ���@    @�@    �	�@     o�@     ��@    @�@     ��@    Љ�@     �@    ��@    ���@    �`�@    `��@    `��@    h#A     �A     @A    `�A    ЅA    ��A    >A    � A    �GA    ��A    �A    P�A    h�A    @A A    �� A    ��A    �oA    �g�@    �,�@    ���@     �@    ���@    @��@     �@     U�@     1�@     �@     X�@     t�@     �@     p�@     �D@        
"
CNN2/weights/summaries/mean�\��
&
CNN2/weights/summaries/stddev_1��=
!
CNN2/weights/summaries/maxf�a>
!
CNN2/weights/summaries/min��m�
�
 CNN2/weights/summaries/histogram*�	   `r�Ϳ   �=�?      �@! +�j8�)�/�-Y@2�
�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.��K+�E��Ͼ['�?�;�ѩ�-�>���%�>�uE����>>�?�s��>�FF�G ?6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?�������:�
              @     �D@     �d@     �h@     �o@     �p@     `o@     �q@     @t@     �t@      r@     t@     pr@      o@     �q@     @n@     �k@     @o@      i@     �f@     `f@     �a@      `@     �^@     @Z@     �\@     �V@     �V@     �S@      R@      E@     �Q@     @S@      O@     �G@     �D@      E@      >@      6@      <@      :@      @@      1@      5@      3@      "@      "@      .@      3@      "@       @      &@      "@      @      @      @      @      @      @      @       @      @       @      @      �?      @      �?      �?      @      �?       @       @              �?               @      �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?      �?      �?      @              �?              @      @      @       @       @       @       @      @       @      @      @      "@      @      @      @       @       @      @       @       @      ,@      (@      ,@      1@      ,@      1@      >@      <@      @@      A@      6@      C@      G@      A@     �B@     �L@      H@     �N@     �R@     �P@      U@      T@      Y@      X@     @[@     �a@     @b@     �c@     �d@     �c@     `g@     @h@      m@     �l@      o@     �q@     �o@     Ps@      r@     �p@     �r@     �t@     pq@      q@     �k@      f@     �`@      <@      �?        
!
CNN2/biases/summaries/meanj��=
%
CNN2/biases/summaries/stddev_1���;
 
CNN2/biases/summaries/max��=
 
CNN2/biases/summaries/min�8�=
�
CNN2/biases/summaries/histogram*�	   ��?   �5�?      @@!   <�	@)��I�cx�?2(� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:(              �?      7@       @        
�
CNN2/activations*�   ��p*@      `A!���
A\A)b��6YqA2�
        �-���q=����>豪}0ڰ>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�������:�
           �'�IA              �?              �?              �?      �?              @      @      �?      @      �?       @      �?      �?       @               @       @      �?      @      @      �?      @      @       @      @      @      @      @      @      @      $@      @      "@      "@      @      &@      .@      (@      $@      $@      *@      5@      1@      5@      1@      :@     �B@      >@     �B@      F@     �G@     �H@      F@     �K@      N@     �P@     �T@      S@      V@     �[@     �_@     �[@     �`@     `b@      d@      f@      h@     �g@     �p@     0p@     �q@     �q@     �t@      v@     y@     �{@      ~@     X�@     ��@      �@     І@     ��@     ��@     h�@     <�@     ܒ@     P�@     ��@     �@     ؚ@     \�@     Р@     ,�@     ��@     ��@     ��@     �@     ��@     �@     t�@     �@     �@     H�@     ù@     �@     ��@    ���@    ���@    ���@    ���@     ��@    ���@     3�@     ��@    �:�@    @1�@    �>�@    ���@     �@    @��@     ��@     '�@     9�@     8�@    ���@    ��@     �@     ��@    �@�@    =�@    pl�@    P��@    ���@    ���@    xJA    0zA    ��A    x�A    ��A    �=	A    ��	A    p�	A    p@
A    P
A    x:
A    X�A    8�A     AA    8�A    PXA    �hA    ��@    @[�@    ��@    p@�@     �@    ���@     ��@     ��@     J�@     ��@     ��@     pu@     �\@      >@      @        
�'
CNN2/batch_norm*�'	   ���    U�$@      `A!��o\$*�@)Y��� `A2�������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ;�"�qʾ
�/eq
Ⱦ�XQ�þ��~��¾�*��ڽ�G&�$��5�"�g������?�ګ�;9��R�����m!#���
�%W���H5�8�t>�i����v>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@�������:�            X99A    ��6A    84A    ��A    �u�@    `�@    ���@    `��@    ���@    ���@    �o�@    ���@    ��@    ���@    ���@    `��@     ��@    @(�@     :�@    ���@    �}�@    @��@    @(�@    ��@    ��@     �@     Y�@     �@    ���@    ��@    �k�@    �q�@    �q�@     �@    �9�@     0�@     ��@     <�@     ��@     ��@     ��@     p�@     B�@     $�@     �@     D�@     �@     ,�@     f�@     ��@     @�@     ��@     |�@     ��@     T�@     <�@     0�@     @�@     �@     ��@     ��@     (�@     �@     ��@     �}@     pz@     y@     �u@     `t@     �s@     �q@     �j@     @l@     �g@     `i@     @b@      a@      _@      ]@     @X@      \@      Z@     �X@     �T@     @T@     �S@      J@      J@     �@@     �J@      B@      G@      @@      :@      <@      6@      &@      3@      4@      0@      ,@      0@      $@      *@      &@      0@      &@      @      @      @      @       @      @      @      @      �?               @              @       @      �?      @      @      �?               @              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?      �?      �?              �?               @      �?              @      �?      �?              �?       @       @       @      @       @      @      @      @      @      @      @      �?      @      @       @       @      @      @      @      �?      @      &@      *@      $@      *@      .@      9@      &@      0@      0@      ;@      ?@      ;@      B@      G@      A@      E@      H@     �K@      Q@     �L@     �P@     @P@     �T@     �Y@      Y@     �^@     @`@     ``@     �b@     `e@     �f@     �i@     �k@      m@     �p@      q@     0s@     0v@      x@     �z@     �|@     �|@     �@     p�@     @�@     Ї@     ��@     ��@     P�@     А@     \�@     ��@     ��@     ��@     �@     ̞@     z�@     z�@     ��@     h�@     `�@     |�@     �@     L�@     �@     R�@     �@     ��@     ��@     ��@     �@    ��@    ���@     h�@    �4�@     s�@     ��@    �9�@    ���@    �d�@    �H�@     ��@    @��@    ���@    @�@     ��@    ���@    `L�@     ��@     ^�@     �@     ��@     ��@    `��@     ��@    PH�@    �t�@    З�@    ���@    �J�@    ���@     ��@    p�@    ���@    �C�@    �^�@    `�@    �C�@     d�@    ��@    ��@    �q�@    �]�@    `��@    @��@    ��@    �`�@    �Z�@    `�@    @��@    @w�@     ��@    �h�@     !�@     t�@     �@     ȑ@     �{@     `g@      H@      *@       @        
"
CNN3/weights/summaries/mean I59
&
CNN3/weights/summaries/stddev_1��=
!
CNN3/weights/summaries/max��d>
!
CNN3/weights/summaries/minK�k�
�
 CNN3/weights/summaries/histogram*�	   `ipͿ   @=��?      �@! Pā �@)����/p@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[��O�ʗ�����Zr[v��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(��uE���⾮��%Ᾱ��?�ګ�;9��R���[�=�k�>��~���>�XQ��>�����>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>>�?�s��>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?�������:�              $@     @T@     pu@     @�@     ��@     ��@     ��@     �@     �@     0�@     (�@      �@     p�@     ��@     �@     ��@     ��@     �@     h�@     p}@     �z@     �v@     �u@     �u@     �q@     �p@      o@     �k@     �f@     �h@      g@     �e@     @c@     �]@      ]@     �Z@     @Y@     �U@     @V@     @U@     @P@     �O@      J@     �J@      J@      A@      =@      B@      A@      ;@      4@      &@      <@      4@      0@      "@      *@      ,@      @      (@      "@      $@      @      @      "@      @      @      @      @      @      @      @      �?      �?       @              @      �?      @      �?       @      @              �?              �?      �?      @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @      �?              �?       @               @      @               @               @      @      @      @      @      @      @      @      @      @      @       @      (@      @       @      @      (@      (@      @      "@      &@      *@      3@      8@      5@      .@      >@      0@      :@      7@      A@     �J@     �B@     �K@     �M@      N@     �Q@      O@     @T@     �Y@     �Y@     @Y@      Z@     `b@     �]@     �c@     `d@      e@     @h@     @l@     �o@     Pq@     �r@     �q@     `v@      y@     �|@      {@     �}@     `�@     ��@     `�@      �@      �@     ��@     Ї@     ��@     ��@     ��@     Ј@     ��@     ��@     �@     8�@     �v@     �[@      @        
!
CNN3/biases/summaries/meanms�=
%
CNN3/biases/summaries/stddev_1�:�;
 
CNN3/biases/summaries/max���=
 
CNN3/biases/summaries/min2��=
�
CNN3/biases/summaries/histogram*�	   @��?   `��?      P@!  ��mN@)HH�w%��?20�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:0              @      2@      C@      @        
�
CNN3/activations*�   ��*,@      PA!���? =A).Cmۺ�PA2�
        �-���q=����>豪}0ڰ>�[�=�k�>��~���>�XQ��>�����>��~]�[�>��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>���%�>�uE����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�������:�
           ��4DA              �?              �?      �?      �?              �?      �?              �?              �?              �?       @      @      @               @      @      @      @      �?      @       @       @       @       @      @      &@      @      @      @      �?      @      @      (@      .@      ,@      @      (@      0@      1@      2@      =@      7@      4@      6@      7@      D@      B@     �H@      B@     �D@      L@      N@      K@      P@      S@      U@      X@     �Z@     @\@     �\@      `@      _@     �b@     �e@     �g@     `l@      m@     �p@     0p@     �r@     `v@     �z@      {@     p@     ��@     `�@     @�@     h�@     ��@     ��@     ��@     0�@     đ@     �@      �@     4�@     ��@     ��@     ��@     ��@     ��@     $�@     �@     ��@     ܪ@     ^�@     r�@     .�@     +�@     "�@     �@     غ@     ͼ@    ��@     Z�@     �@    �G�@    �l�@    �j�@     ��@     ��@    ���@     �@     ��@    ���@    @��@    �{�@    ��@    ���@     ��@    @��@    @P�@    ���@     !�@     ��@    �A�@    ���@    �[�@    @��@    ���@    ��@    ���@    �}�@     }�@    `�@     ��@     ��@    ���@    @��@     G�@    ��@    @��@    �+�@    @C�@    �L�@     ��@     ��@     ��@     ��@     &�@     h�@     X�@     `l@      T@      ?@       @      @        
�%
CNN3/batch_norm*�%	   �7�   �-@      PA!?]��X�A)��^kZPA2��1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f���侙ѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ����ž�XQ�þ��~��¾0�6�/n�>5�"�g��>�*��ڽ>�[�=�k�>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>jqs&\��>��~]�[�>��>M|K�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�������:�            8�A    ��@A    dZA    @��@    @��@     -�@    ���@    ��@    �e�@    ���@     !�@     0�@    ���@    ��@    ��@    �<�@    ���@     ��@     1�@     Ѹ@     O�@     b�@     I�@     ��@     |�@     Z�@     Ȩ@     ��@      �@     2�@     �@     8�@     �@     p�@     L�@     ĕ@     p�@      �@     D�@     x�@     @�@     h�@     ��@     ��@     ��@     ��@     �}@     �|@     �x@     �v@      s@     0r@     �p@     `k@     �k@     �f@     �e@     @b@     `b@     �a@     @^@     �Z@     �X@     @U@     �T@     @S@      N@     �I@     �P@     �L@      C@     �G@      ;@      ;@      ?@      5@      7@      :@      2@      2@      .@      <@      $@      1@      ,@      *@      0@      *@      $@      @      "@      $@      @      (@      @      @      @      @      @      @      �?              @       @       @              �?      �?       @       @              �?      �?              �?               @      �?      �?              �?      �?      �?              �?      �?              �?              �?              �?              �?      �?              �?      �?               @              �?      �?               @              �?      @      �?      �?       @      @               @              @               @      @       @      @       @      @      @      "@      @       @      @      "@      @      ,@      &@      0@      *@      "@      3@      .@      3@      4@      7@      1@      >@      =@      ;@     �A@      E@     �D@     �G@     �G@      K@     �P@     �P@     �T@     �T@     �Y@     �\@     �`@      _@      _@     �a@     @f@     �g@     `k@      k@      p@     �o@     0q@     `s@     `u@     �x@      z@     �|@     @     0�@     ��@     ��@     @�@     ��@     ��@     (�@     ��@     ��@     ��@     @�@     \�@     ��@     �@     ��@     ��@     ��@     �@     Ш@     D�@     ��@     8�@     ��@     '�@     ,�@     ��@     ��@     ,�@     '�@     ��@     _�@    �N�@     ��@    ���@    ���@    �d�@     +�@     	�@    @A�@    ���@    ���@    ���@    �W�@    ��@    @��@    @��@    ` �@    ���@    `�@    @�@    �|�@    ���@    �p�@     ��@    ���@    �7�@    �Q�@    ���@    ���@    ���@    ���@     H�@     ��@     B�@    @��@    � �@     ��@     1�@     N�@     K�@     )�@     ��@     �@     ��@     ��@     �q@     �\@     �H@      0@      @      �?        
 
accuracy_streaming_mean_1��?Fę�