       �K"	   L��Abrain.Event:2�=���     -��$	�n	L��A"��
�
input/imagesPlaceholder"/device:GPU:0*
dtype0*/
_output_shapes
:���������@@*$
shape:���������@@
�
input/correct_labelsPlaceholder"/device:GPU:0*
shape:���������*
dtype0*'
_output_shapes
:���������
g
input/training_flagPlaceholder"/device:GPU:0*
dtype0
*
_output_shapes
:*
shape:
�
#CNN1/weights/truncated_normal/shapeConst"/device:GPU:0*%
valueB"            *
dtype0*
_output_shapes
:
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
-CNN1/weights/truncated_normal/TruncatedNormalTruncatedNormal#CNN1/weights/truncated_normal/shape"/device:GPU:0*&
_output_shapes
:*
seed2 *

seed *
T0*
dtype0
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
CNN1/weights/Variable/readIdentityCNN1/weights/Variable"/device:GPU:0*(
_class
loc:@CNN1/weights/Variable*&
_output_shapes
:*
T0
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
CNN1/weights/summaries/meanScalarSummary CNN1/weights/summaries/mean/tagsCNN1/weights/summaries/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
!CNN1/weights/summaries/stddev/subSubCNN1/weights/Variable/readCNN1/weights/summaries/Mean"/device:GPU:0*
T0*&
_output_shapes
:
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
"CNN1/weights/summaries/stddev/MeanMean$CNN1/weights/summaries/stddev/Square#CNN1/weights/summaries/stddev/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
CNN1/weights/summaries/stddev_1ScalarSummary$CNN1/weights/summaries/stddev_1/tags"CNN1/weights/summaries/stddev/Sqrt"/device:GPU:0*
T0*
_output_shapes
: 
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
$CNN1/weights/summaries/range_1/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN1/weights/summaries/range_1Range$CNN1/weights/summaries/range_1/startCNN1/weights/summaries/Rank_1$CNN1/weights/summaries/range_1/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/weights/summaries/MaxMaxCNN1/weights/Variable/readCNN1/weights/summaries/range_1"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN1/weights/summaries/max/tagsConst"/device:GPU:0*+
value"B  BCNN1/weights/summaries/max*
dtype0*
_output_shapes
: 
�
CNN1/weights/summaries/maxScalarSummaryCNN1/weights/summaries/max/tagsCNN1/weights/summaries/Max"/device:GPU:0*
T0*
_output_shapes
: 
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
CNN1/weights/summaries/MinMinCNN1/weights/Variable/readCNN1/weights/summaries/range_2"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
CNN1/weights/summaries/min/tagsConst"/device:GPU:0*+
value"B  BCNN1/weights/summaries/min*
dtype0*
_output_shapes
: 
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
CNN1/biases/ConstConst"/device:GPU:0*
_output_shapes
:*
valueB*���=*
dtype0
�
CNN1/biases/Variable
VariableV2"/device:GPU:0*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
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
CNN1/biases/summaries/rangeRange!CNN1/biases/summaries/range/startCNN1/biases/summaries/Rank!CNN1/biases/summaries/range/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
�
CNN1/biases/summaries/MeanMeanCNN1/biases/Variable/readCNN1/biases/summaries/range"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN1/biases/summaries/mean/tagsConst"/device:GPU:0*
_output_shapes
: *+
value"B  BCNN1/biases/summaries/mean*
dtype0
�
CNN1/biases/summaries/meanScalarSummaryCNN1/biases/summaries/mean/tagsCNN1/biases/summaries/Mean"/device:GPU:0*
T0*
_output_shapes
: 
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
!CNN1/biases/summaries/stddev/SqrtSqrt!CNN1/biases/summaries/stddev/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
#CNN1/biases/summaries/stddev_1/tagsConst"/device:GPU:0*
_output_shapes
: */
value&B$ BCNN1/biases/summaries/stddev_1*
dtype0
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
CNN1/biases/summaries/range_1Range#CNN1/biases/summaries/range_1/startCNN1/biases/summaries/Rank_1#CNN1/biases/summaries/range_1/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
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
CNN1/biases/summaries/range_2Range#CNN1/biases/summaries/range_2/startCNN1/biases/summaries/Rank_2#CNN1/biases/summaries/range_2/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
�
CNN1/biases/summaries/MinMinCNN1/biases/Variable/readCNN1/biases/summaries/range_2"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN1/biases/summaries/min/tagsConst"/device:GPU:0**
value!B BCNN1/biases/summaries/min*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/minScalarSummaryCNN1/biases/summaries/min/tagsCNN1/biases/summaries/Min"/device:GPU:0*
T0*
_output_shapes
: 
�
#CNN1/biases/summaries/histogram/tagConst"/device:GPU:0*
_output_shapes
: *0
value'B% BCNN1/biases/summaries/histogram*
dtype0
�
CNN1/biases/summaries/histogramHistogramSummary#CNN1/biases/summaries/histogram/tagCNN1/biases/Variable/read"/device:GPU:0*
T0*
_output_shapes
: 
�
CNN1/Conv2DConv2Dinput/imagesCNN1/weights/Variable/read"/device:GPU:0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������@@*
T0
�
CNN1/addAddCNN1/Conv2DCNN1/biases/Variable/read"/device:GPU:0*
T0*/
_output_shapes
:���������@@
d
	CNN1/ReluReluCNN1/add"/device:GPU:0*/
_output_shapes
:���������@@*
T0
t
CNN1/activations/tagConst"/device:GPU:0*
_output_shapes
: *!
valueB BCNN1/activations*
dtype0
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
VariableV2"/device:GPU:0*
_output_shapes
:@*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:@*
dtype0
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
batch_normalization/gamma/readIdentitybatch_normalization/gamma"/device:GPU:0*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
�
*batch_normalization/beta/Initializer/zerosConst*
_output_shapes
:@*+
_class!
loc:@batch_normalization/beta*
valueB@*    *
dtype0
�
batch_normalization/beta
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container *
shape:@
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
1batch_normalization/moving_mean/Initializer/zerosConst*2
_class(
&$loc:@batch_normalization/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
�
batch_normalization/moving_mean
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:@*
shared_name *2
_class(
&$loc:@batch_normalization/moving_mean*
	container *
shape:@
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
4batch_normalization/moving_variance/Initializer/onesConst*6
_class,
*(loc:@batch_normalization/moving_variance*
valueB@*  �?*
dtype0*
_output_shapes
:@
�
#batch_normalization/moving_variance
VariableV2"/device:GPU:0*
shared_name *6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
*batch_normalization/moving_variance/AssignAssign#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones"/device:GPU:0*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
(batch_normalization/moving_variance/readIdentity#batch_normalization/moving_variance"/device:GPU:0*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
�
$CNN1/batch_normalization/cond/SwitchSwitchinput/training_flaginput/training_flag"/device:GPU:0*
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
&CNN1/batch_normalization/cond/switch_fIdentity$CNN1/batch_normalization/cond/Switch"/device:GPU:0*
_output_shapes
:*
T0

x
%CNN1/batch_normalization/cond/pred_idIdentityinput/training_flag"/device:GPU:0*
T0
*
_output_shapes
:
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
5CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1Switchbatch_normalization/gamma/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
:@:@*
T0*,
_class"
 loc:@batch_normalization/gamma
�
5CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization/beta/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
:@:@*
T0*+
_class!
loc:@batch_normalization/beta
�
,CNN1/batch_normalization/cond/FusedBatchNormFusedBatchNorm5CNN1/batch_normalization/cond/FusedBatchNorm/Switch:17CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1:17CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2:1#CNN1/batch_normalization/cond/Const%CNN1/batch_normalization/cond/Const_1"/device:GPU:0*
data_formatNCHW*G
_output_shapes5
3:���������@@:@:@:@:@*
is_training(*
epsilon%o�:*
T0
�
5CNN1/batch_normalization/cond/FusedBatchNorm_1/SwitchSwitch	CNN1/Relu%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*J
_output_shapes8
6:���������@@:���������@@*
T0*
_class
loc:@CNN1/Relu
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
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch$batch_normalization/moving_mean/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*
T0*2
_class(
&$loc:@batch_normalization/moving_mean* 
_output_shapes
:@:@
�
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch(batch_normalization/moving_variance/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*
T0*6
_class,
*(loc:@batch_normalization/moving_variance* 
_output_shapes
:@:@
�
.CNN1/batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm5CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_27CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*G
_output_shapes5
3:���������@@:@:@:@:@*
is_training( *
epsilon%o�:*
T0*
data_formatNCHW
�
#CNN1/batch_normalization/cond/MergeMerge.CNN1/batch_normalization/cond/FusedBatchNorm_1,CNN1/batch_normalization/cond/FusedBatchNorm"/device:GPU:0*
T0*
N*1
_output_shapes
:���������@@: 
�
%CNN1/batch_normalization/cond/Merge_1Merge0CNN1/batch_normalization/cond/FusedBatchNorm_1:1.CNN1/batch_normalization/cond/FusedBatchNorm:1"/device:GPU:0*
T0*
N*
_output_shapes

:@: 
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
+CNN1/batch_normalization/ExpandDims_1/inputConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
z
)CNN1/batch_normalization/ExpandDims_1/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
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
 CNN1/batch_normalization/ReshapeReshapeinput/training_flag&CNN1/batch_normalization/Reshape/shape"/device:GPU:0*
T0
*
Tshape0*
_output_shapes
:
�
CNN1/batch_normalization/SelectSelect CNN1/batch_normalization/Reshape#CNN1/batch_normalization/ExpandDims%CNN1/batch_normalization/ExpandDims_1"/device:GPU:0*
T0*
_output_shapes
:
�
 CNN1/batch_normalization/SqueezeSqueezeCNN1/batch_normalization/Select"/device:GPU:0*
_output_shapes
: *
squeeze_dims
 *
T0
�
-CNN1/batch_normalization/AssignMovingAvg/readIdentitybatch_normalization/moving_mean"/device:GPU:0*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
�
,CNN1/batch_normalization/AssignMovingAvg/SubSub-CNN1/batch_normalization/AssignMovingAvg/read%CNN1/batch_normalization/cond/Merge_1"/device:GPU:0*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
�
,CNN1/batch_normalization/AssignMovingAvg/MulMul,CNN1/batch_normalization/AssignMovingAvg/Sub CNN1/batch_normalization/Squeeze"/device:GPU:0*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
�
(CNN1/batch_normalization/AssignMovingAvg	AssignSubbatch_normalization/moving_mean,CNN1/batch_normalization/AssignMovingAvg/Mul"/device:GPU:0*
_output_shapes
:@*
use_locking( *
T0*2
_class(
&$loc:@batch_normalization/moving_mean
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
.CNN1/batch_normalization/AssignMovingAvg_1/MulMul.CNN1/batch_normalization/AssignMovingAvg_1/Sub CNN1/batch_normalization/Squeeze"/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
*CNN1/batch_normalization/AssignMovingAvg_1	AssignSub#batch_normalization/moving_variance.CNN1/batch_normalization/AssignMovingAvg_1/Mul"/device:GPU:0*
use_locking( *
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
r
CNN1/batch_norm/tagConst"/device:GPU:0*
_output_shapes
: * 
valueB BCNN1/batch_norm*
dtype0
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
!CNN2/weights/truncated_normal/mulMul-CNN2/weights/truncated_normal/TruncatedNormal$CNN2/weights/truncated_normal/stddev"/device:GPU:0*
T0*&
_output_shapes
: 
�
CNN2/weights/truncated_normalAdd!CNN2/weights/truncated_normal/mul"CNN2/weights/truncated_normal/mean"/device:GPU:0*
T0*&
_output_shapes
: 
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
CNN2/weights/Variable/AssignAssignCNN2/weights/VariableCNN2/weights/truncated_normal"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN2/weights/Variable*
validate_shape(*&
_output_shapes
: 
�
CNN2/weights/Variable/readIdentityCNN2/weights/Variable"/device:GPU:0*(
_class
loc:@CNN2/weights/Variable*&
_output_shapes
: *
T0
l
CNN2/weights/summaries/RankConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
s
"CNN2/weights/summaries/range/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
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
$CNN2/weights/summaries/stddev/SquareSquare!CNN2/weights/summaries/stddev/sub"/device:GPU:0*&
_output_shapes
: *
T0
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
CNN2/weights/summaries/maxScalarSummaryCNN2/weights/summaries/max/tagsCNN2/weights/summaries/Max"/device:GPU:0*
_output_shapes
: *
T0
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
$CNN2/weights/summaries/range_2/deltaConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
CNN2/weights/summaries/range_2Range$CNN2/weights/summaries/range_2/startCNN2/weights/summaries/Rank_2$CNN2/weights/summaries/range_2/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN2/weights/summaries/MinMinCNN2/weights/Variable/readCNN2/weights/summaries/range_2"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN2/weights/summaries/min/tagsConst"/device:GPU:0*
_output_shapes
: *+
value"B  BCNN2/weights/summaries/min*
dtype0
�
CNN2/weights/summaries/minScalarSummaryCNN2/weights/summaries/min/tagsCNN2/weights/summaries/Min"/device:GPU:0*
_output_shapes
: *
T0
�
$CNN2/weights/summaries/histogram/tagConst"/device:GPU:0*1
value(B& B CNN2/weights/summaries/histogram*
dtype0*
_output_shapes
: 
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
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
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
!CNN2/biases/summaries/range/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
r
!CNN2/biases/summaries/range/deltaConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
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
!CNN2/biases/summaries/stddev/MeanMean#CNN2/biases/summaries/stddev/Square"CNN2/biases/summaries/stddev/Const"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
|
!CNN2/biases/summaries/stddev/SqrtSqrt!CNN2/biases/summaries/stddev/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
#CNN2/biases/summaries/stddev_1/tagsConst"/device:GPU:0*/
value&B$ BCNN2/biases/summaries/stddev_1*
dtype0*
_output_shapes
: 
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
#CNN2/biases/summaries/range_1/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
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
#CNN2/biases/summaries/range_2/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
t
#CNN2/biases/summaries/range_2/deltaConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
CNN2/biases/summaries/range_2Range#CNN2/biases/summaries/range_2/startCNN2/biases/summaries/Rank_2#CNN2/biases/summaries/range_2/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN2/biases/summaries/MinMinCNN2/biases/Variable/readCNN2/biases/summaries/range_2"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN2/biases/summaries/min/tagsConst"/device:GPU:0**
value!B BCNN2/biases/summaries/min*
dtype0*
_output_shapes
: 
�
CNN2/biases/summaries/minScalarSummaryCNN2/biases/summaries/min/tagsCNN2/biases/summaries/Min"/device:GPU:0*
_output_shapes
: *
T0
�
#CNN2/biases/summaries/histogram/tagConst"/device:GPU:0*
_output_shapes
: *0
value'B% BCNN2/biases/summaries/histogram*
dtype0
�
CNN2/biases/summaries/histogramHistogramSummary#CNN2/biases/summaries/histogram/tagCNN2/biases/Variable/read"/device:GPU:0*
_output_shapes
: *
T0
�
CNN2/Conv2DConv2D#CNN1/batch_normalization/cond/MergeCNN2/weights/Variable/read"/device:GPU:0*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������   
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
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones"/device:GPU:0*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(
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
VariableV2"/device:GPU:0*
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container *
shape: *
dtype0*
_output_shapes
: 
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
batch_normalization_1/beta/readIdentitybatch_normalization_1/beta"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: 
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
VariableV2"/device:GPU:0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
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
6batch_normalization_1/moving_variance/Initializer/onesConst*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB *  �?*
dtype0
�
%batch_normalization_1/moving_variance
VariableV2"/device:GPU:0*
shared_name *8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: 
�
,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones"/device:GPU:0*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes
: 
�
*batch_normalization_1/moving_variance/readIdentity%batch_normalization_1/moving_variance"/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
$CNN2/batch_normalization/cond/SwitchSwitchinput/training_flaginput/training_flag"/device:GPU:0*
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
&CNN2/batch_normalization/cond/switch_fIdentity$CNN2/batch_normalization/cond/Switch"/device:GPU:0*
_output_shapes
:*
T0

x
%CNN2/batch_normalization/cond/pred_idIdentityinput/training_flag"/device:GPU:0*
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
3CNN2/batch_normalization/cond/FusedBatchNorm/SwitchSwitch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
_class
loc:@CNN2/Relu*J
_output_shapes8
6:���������   :���������   *
T0
�
5CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1Switch batch_normalization_1/gamma/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
: : 
�
5CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
: : 
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
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
: : *
T0
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_1/moving_mean/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean* 
_output_shapes
: : 
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_1/moving_variance/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance* 
_output_shapes
: : 
�
.CNN2/batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm5CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_27CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
T0*
data_formatNCHW*G
_output_shapes5
3:���������   : : : : *
is_training( *
epsilon%o�:
�
#CNN2/batch_normalization/cond/MergeMerge.CNN2/batch_normalization/cond/FusedBatchNorm_1,CNN2/batch_normalization/cond/FusedBatchNorm"/device:GPU:0*
T0*
N*1
_output_shapes
:���������   : 
�
%CNN2/batch_normalization/cond/Merge_1Merge0CNN2/batch_normalization/cond/FusedBatchNorm_1:1.CNN2/batch_normalization/cond/FusedBatchNorm:1"/device:GPU:0*
T0*
N*
_output_shapes

: : 
�
%CNN2/batch_normalization/cond/Merge_2Merge0CNN2/batch_normalization/cond/FusedBatchNorm_1:2.CNN2/batch_normalization/cond/FusedBatchNorm:2"/device:GPU:0*
T0*
N*
_output_shapes

: : 
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
)CNN2/batch_normalization/ExpandDims_1/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
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
 CNN2/batch_normalization/ReshapeReshapeinput/training_flag&CNN2/batch_normalization/Reshape/shape"/device:GPU:0*
Tshape0*
_output_shapes
:*
T0

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
,CNN2/batch_normalization/AssignMovingAvg/MulMul,CNN2/batch_normalization/AssignMovingAvg/Sub CNN2/batch_normalization/Squeeze"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
�
(CNN2/batch_normalization/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean,CNN2/batch_normalization/AssignMovingAvg/Mul"/device:GPU:0*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
/CNN2/batch_normalization/AssignMovingAvg_1/readIdentity%batch_normalization_1/moving_variance"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
�
.CNN2/batch_normalization/AssignMovingAvg_1/SubSub/CNN2/batch_normalization/AssignMovingAvg_1/read%CNN2/batch_normalization/cond/Merge_2"/device:GPU:0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: *
T0
�
.CNN2/batch_normalization/AssignMovingAvg_1/MulMul.CNN2/batch_normalization/AssignMovingAvg_1/Sub CNN2/batch_normalization/Squeeze"/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
*CNN2/batch_normalization/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance.CNN2/batch_normalization/AssignMovingAvg_1/Mul"/device:GPU:0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: *
use_locking( *
T0
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
VariableV2"/device:GPU:0*
shared_name *
dtype0*&
_output_shapes
: @*
	container *
shape: @
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
CNN3/weights/summaries/MeanMeanCNN3/weights/Variable/readCNN3/weights/summaries/range"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
!CNN3/weights/summaries/stddev/subSubCNN3/weights/Variable/readCNN3/weights/summaries/Mean"/device:GPU:0*&
_output_shapes
: @*
T0
�
$CNN3/weights/summaries/stddev/SquareSquare!CNN3/weights/summaries/stddev/sub"/device:GPU:0*&
_output_shapes
: @*
T0
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
$CNN3/weights/summaries/range_1/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
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
CNN3/weights/summaries/MinMinCNN3/weights/Variable/readCNN3/weights/summaries/range_2"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN3/weights/summaries/min/tagsConst"/device:GPU:0*+
value"B  BCNN3/weights/summaries/min*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/minScalarSummaryCNN3/weights/summaries/min/tagsCNN3/weights/summaries/Min"/device:GPU:0*
_output_shapes
: *
T0
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
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
�
CNN3/biases/Variable/AssignAssignCNN3/biases/VariableCNN3/biases/Const"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN3/biases/Variable*
validate_shape(*
_output_shapes
:@
�
CNN3/biases/Variable/readIdentityCNN3/biases/Variable"/device:GPU:0*
T0*'
_class
loc:@CNN3/biases/Variable*
_output_shapes
:@
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
CNN3/biases/summaries/MeanMeanCNN3/biases/Variable/readCNN3/biases/summaries/range"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN3/biases/summaries/mean/tagsConst"/device:GPU:0*
_output_shapes
: *+
value"B  BCNN3/biases/summaries/mean*
dtype0
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
!CNN3/biases/summaries/stddev/MeanMean#CNN3/biases/summaries/stddev/Square"CNN3/biases/summaries/stddev/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
CNN3/biases/summaries/stddev_1ScalarSummary#CNN3/biases/summaries/stddev_1/tags!CNN3/biases/summaries/stddev/Sqrt"/device:GPU:0*
_output_shapes
: *
T0
m
CNN3/biases/summaries/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
t
#CNN3/biases/summaries/range_1/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
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
CNN3/biases/summaries/MaxMaxCNN3/biases/Variable/readCNN3/biases/summaries/range_1"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
CNN3/biases/summaries/min/tagsConst"/device:GPU:0**
value!B BCNN3/biases/summaries/min*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/minScalarSummaryCNN3/biases/summaries/min/tagsCNN3/biases/summaries/Min"/device:GPU:0*
_output_shapes
: *
T0
�
#CNN3/biases/summaries/histogram/tagConst"/device:GPU:0*0
value'B% BCNN3/biases/summaries/histogram*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/histogramHistogramSummary#CNN3/biases/summaries/histogram/tagCNN3/biases/Variable/read"/device:GPU:0*
_output_shapes
: *
T0
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
,batch_normalization_2/gamma/Initializer/onesConst*
_output_shapes
:*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*  �?*
dtype0
�
batch_normalization_2/gamma
VariableV2"/device:GPU:0*
shape:*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container 
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
 batch_normalization_2/gamma/readIdentitybatch_normalization_2/gamma"/device:GPU:0*
_output_shapes
:*
T0*.
_class$
" loc:@batch_normalization_2/gamma
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
VariableV2"/device:GPU:0*
_output_shapes
:*
shared_name *-
_class#
!loc:@batch_normalization_2/beta*
	container *
shape:*
dtype0
�
!batch_normalization_2/beta/AssignAssignbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(
�
batch_normalization_2/beta/readIdentitybatch_normalization_2/beta"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:
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
VariableV2"/device:GPU:0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean"/device:GPU:0*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
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
VariableV2"/device:GPU:0*
_output_shapes
:*
shared_name *8
_class.
,*loc:@batch_normalization_2/moving_variance*
	container *
shape:*
dtype0
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
*batch_normalization_2/moving_variance/readIdentity%batch_normalization_2/moving_variance"/device:GPU:0*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
$CNN3/batch_normalization/cond/SwitchSwitchinput/training_flaginput/training_flag"/device:GPU:0*
T0
*
_output_shapes

::
�
&CNN3/batch_normalization/cond/switch_tIdentity&CNN3/batch_normalization/cond/Switch:1"/device:GPU:0*
T0
*
_output_shapes
:
�
&CNN3/batch_normalization/cond/switch_fIdentity$CNN3/batch_normalization/cond/Switch"/device:GPU:0*
T0
*
_output_shapes
:
x
%CNN3/batch_normalization/cond/pred_idIdentityinput/training_flag"/device:GPU:0*
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
5CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_2/gamma
�
5CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
::
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
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_2/gamma* 
_output_shapes
::
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
::*
T0*-
_class#
!loc:@batch_normalization_2/beta
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_2/moving_mean/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*4
_class*
(&loc:@batch_normalization_2/moving_mean* 
_output_shapes
::*
T0
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_2/moving_variance/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*8
_class.
,*loc:@batch_normalization_2/moving_variance* 
_output_shapes
::*
T0
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
%CNN3/batch_normalization/cond/Merge_1Merge0CNN3/batch_normalization/cond/FusedBatchNorm_1:1.CNN3/batch_normalization/cond/FusedBatchNorm:1"/device:GPU:0*
_output_shapes

:: *
T0*
N
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
ExpandDims+CNN3/batch_normalization/ExpandDims_1/input)CNN3/batch_normalization/ExpandDims_1/dim"/device:GPU:0*
_output_shapes
:*

Tdim0*
T0

&CNN3/batch_normalization/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
 CNN3/batch_normalization/ReshapeReshapeinput/training_flag&CNN3/batch_normalization/Reshape/shape"/device:GPU:0*
T0
*
Tshape0*
_output_shapes
:
�
CNN3/batch_normalization/SelectSelect CNN3/batch_normalization/Reshape#CNN3/batch_normalization/ExpandDims%CNN3/batch_normalization/ExpandDims_1"/device:GPU:0*
T0*
_output_shapes
:
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
,CNN3/batch_normalization/AssignMovingAvg/SubSub-CNN3/batch_normalization/AssignMovingAvg/read%CNN3/batch_normalization/cond/Merge_1"/device:GPU:0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:*
T0
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
.CNN3/batch_normalization/AssignMovingAvg_1/SubSub/CNN3/batch_normalization/AssignMovingAvg_1/read%CNN3/batch_normalization/cond/Merge_2"/device:GPU:0*
_output_shapes
:*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
�
.CNN3/batch_normalization/AssignMovingAvg_1/MulMul.CNN3/batch_normalization/AssignMovingAvg_1/Sub CNN3/batch_normalization/Squeeze"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
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
CNN3/batch_normHistogramSummaryCNN3/batch_norm/tag#CNN3/batch_normalization/cond/Merge"/device:GPU:0*
_output_shapes
: *
T0
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
FC1/truncated_normal/stddevConst"/device:GPU:0*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
$FC1/truncated_normal/TruncatedNormalTruncatedNormalFC1/truncated_normal/shape"/device:GPU:0*

seed *
T0*
dtype0*!
_output_shapes
:���*
seed2 
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
VariableV2"/device:GPU:0*
shared_name *
dtype0*!
_output_shapes
:���*
	container *
shape:���
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
FC1/Reshape/shapeConst"/device:GPU:0*
_output_shapes
:*
valueB"���� @  *
dtype0
�
FC1/ReshapeReshape#CNN3/batch_normalization/cond/MergeFC1/Reshape/shape"/device:GPU:0*)
_output_shapes
:�����������*
T0*
Tshape0
�

FC1/MatMulMatMulFC1/ReshapeFC1/Variable/read"/device:GPU:0*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
q
FC1/addAdd
FC1/MatMulFC1/Variable_1/read"/device:GPU:0*
T0*(
_output_shapes
:����������
[
FC1/ReluReluFC1/add"/device:GPU:0*(
_output_shapes
:����������*
T0
�
,batch_normalization_3/gamma/Initializer/onesConst*
_output_shapes	
:�*.
_class$
" loc:@batch_normalization_3/gamma*
valueB�*  �?*
dtype0
�
batch_normalization_3/gamma
VariableV2"/device:GPU:0*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container 
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
VariableV2"/device:GPU:0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container *
shape:�*
dtype0
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
(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
validate_shape(*
_output_shapes	
:�
�
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean"/device:GPU:0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:�*
T0
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
,batch_normalization_3/moving_variance/AssignAssign%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones"/device:GPU:0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
*batch_normalization_3/moving_variance/readIdentity%batch_normalization_3/moving_variance"/device:GPU:0*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
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
ExpandDims&batch_normalization_3/moving_mean/read(FC1/batch_normalization/ExpandDims_1/dim"/device:GPU:0*
T0*
_output_shapes
:	�*

Tdim0
~
%FC1/batch_normalization/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
FC1/batch_normalization/ReshapeReshapeinput/training_flag%FC1/batch_normalization/Reshape/shape"/device:GPU:0*
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
FC1/batch_normalization/SqueezeSqueezeFC1/batch_normalization/Select"/device:GPU:0*
T0*
_output_shapes	
:�*
squeeze_dims
 
y
(FC1/batch_normalization/ExpandDims_2/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
$FC1/batch_normalization/ExpandDims_2
ExpandDims)FC1/batch_normalization/moments/Squeeze_1(FC1/batch_normalization/ExpandDims_2/dim"/device:GPU:0*
_output_shapes
:	�*

Tdim0*
T0
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
!FC1/batch_normalization/Reshape_1Reshapeinput/training_flag'FC1/batch_normalization/Reshape_1/shape"/device:GPU:0*
_output_shapes
:*
T0
*
Tshape0
�
 FC1/batch_normalization/Select_1Select!FC1/batch_normalization/Reshape_1$FC1/batch_normalization/ExpandDims_2$FC1/batch_normalization/ExpandDims_3"/device:GPU:0*
T0*
_output_shapes
:	�
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
ExpandDims*FC1/batch_normalization/ExpandDims_4/input(FC1/batch_normalization/ExpandDims_4/dim"/device:GPU:0*
_output_shapes
:*

Tdim0*
T0
~
*FC1/batch_normalization/ExpandDims_5/inputConst"/device:GPU:0*
valueB
 *  �?*
dtype0*
_output_shapes
: 
y
(FC1/batch_normalization/ExpandDims_5/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
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
!FC1/batch_normalization/Reshape_2Reshapeinput/training_flag'FC1/batch_normalization/Reshape_2/shape"/device:GPU:0*
T0
*
Tshape0*
_output_shapes
:
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
-FC1/batch_normalization/AssignMovingAvg/sub/xConst"/device:GPU:0*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
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
-FC1/batch_normalization/AssignMovingAvg_1/subSub/FC1/batch_normalization/AssignMovingAvg_1/sub/x!FC1/batch_normalization/Squeeze_2"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
: 
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
)FC1/batch_normalization/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance-FC1/batch_normalization/AssignMovingAvg_1/mul"/device:GPU:0*
_output_shapes	
:�*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
{
'FC1/batch_normalization/batchnorm/add/yConst"/device:GPU:0*
valueB
 *o�:*
dtype0*
_output_shapes
: 
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
'FC1/batch_normalization/batchnorm/add_1Add'FC1/batch_normalization/batchnorm/mul_1%FC1/batch_normalization/batchnorm/sub"/device:GPU:0*(
_output_shapes
:����������*
T0
e
dropout/keep_probPlaceholder"/device:GPU:0*
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
Readout/truncated_normal/stddevConst"/device:GPU:0*
_output_shapes
: *
valueB
 *���=*
dtype0
�
(Readout/truncated_normal/TruncatedNormalTruncatedNormalReadout/truncated_normal/shape"/device:GPU:0*
T0*
dtype0*
_output_shapes
:	�*
seed2 *

seed 
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
VariableV2"/device:GPU:0*
shape:	�*
shared_name *
dtype0*
_output_shapes
:	�*
	container 
�
Readout/Variable/AssignAssignReadout/VariableReadout/truncated_normal"/device:GPU:0*
_output_shapes
:	�*
use_locking(*
T0*#
_class
loc:@Readout/Variable*
validate_shape(
�
Readout/Variable/readIdentityReadout/Variable"/device:GPU:0*
_output_shapes
:	�*
T0*#
_class
loc:@Readout/Variable
i
Readout/ConstConst"/device:GPU:0*
_output_shapes
:*
valueB*���=*
dtype0
�
Readout/Variable_1
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
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
Readout/Variable_1/readIdentityReadout/Variable_1"/device:GPU:0*%
_class
loc:@Readout/Variable_1*
_output_shapes
:*
T0
�
Readout/MatMulMatMul'FC1/batch_normalization/batchnorm/add_1Readout/Variable/read"/device:GPU:0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
Readout/predicted_labelsAddReadout/MatMulReadout/Variable_1/read"/device:GPU:0*
T0*'
_output_shapes
:���������
t
#Readout/predicted_results/dimensionConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
Readout/predicted_resultsArgMaxReadout/predicted_labels#Readout/predicted_results/dimension"/device:GPU:0*
T0*
output_type0	*#
_output_shapes
:���������*

Tidx0
i
cross_entropy_total/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
cross_entropy_total/ShapeShapeReadout/predicted_labels"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
k
cross_entropy_total/Rank_1Const"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
cross_entropy_total/Shape_1ShapeReadout/predicted_labels"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
j
cross_entropy_total/Sub/yConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
cross_entropy_total/SubSubcross_entropy_total/Rank_1cross_entropy_total/Sub/y"/device:GPU:0*
T0*
_output_shapes
: 
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
cross_entropy_total/concat/axisConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
cross_entropy_total/concatConcatV2#cross_entropy_total/concat/values_0cross_entropy_total/Slicecross_entropy_total/concat/axis"/device:GPU:0*
T0*
N*
_output_shapes
:*

Tidx0
�
cross_entropy_total/ReshapeReshapeReadout/predicted_labelscross_entropy_total/concat"/device:GPU:0*
T0*
Tshape0*0
_output_shapes
:������������������
k
cross_entropy_total/Rank_2Const"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
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
!cross_entropy_total/Slice_1/beginPackcross_entropy_total/Sub_1"/device:GPU:0*

axis *
N*
_output_shapes
:*
T0
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
cross_entropy_total/Reshape_1Reshapeinput/correct_labelscross_entropy_total/concat_1"/device:GPU:0*0
_output_shapes
:������������������*
T0*
Tshape0
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
cross_entropy_total/Sub_2Subcross_entropy_total/Rankcross_entropy_total/Sub_2/y"/device:GPU:0*
T0*
_output_shapes
: 
z
!cross_entropy_total/Slice_2/beginConst"/device:GPU:0*
_output_shapes
:*
valueB: *
dtype0
�
 cross_entropy_total/Slice_2/sizePackcross_entropy_total/Sub_2"/device:GPU:0*
_output_shapes
:*
T0*

axis *
N
�
cross_entropy_total/Slice_2Slicecross_entropy_total/Shape!cross_entropy_total/Slice_2/begin cross_entropy_total/Slice_2/size"/device:GPU:0*
Index0*
T0*#
_output_shapes
:���������
�
cross_entropy_total/Reshape_2Reshape1cross_entropy_total/SoftmaxCrossEntropyWithLogitscross_entropy_total/Slice_2"/device:GPU:0*
T0*
Tshape0*#
_output_shapes
:���������
r
cross_entropy_total/ConstConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
cross_entropy_total/MeanMeancross_entropy_total/Reshape_2cross_entropy_total/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
;train/gradients/cross_entropy_total/Mean_grad/Reshape/shapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/cross_entropy_total/Mean_grad/ReshapeReshapetrain/gradients/Fill;train/gradients/cross_entropy_total/Mean_grad/Reshape/shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:
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
2train/gradients/cross_entropy_total/Mean_grad/ProdProd5train/gradients/cross_entropy_total/Mean_grad/Shape_13train/gradients/cross_entropy_total/Mean_grad/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
_output_shapes
: 
�
5train/gradients/cross_entropy_total/Mean_grad/Const_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
4train/gradients/cross_entropy_total/Mean_grad/Prod_1Prod5train/gradients/cross_entropy_total/Mean_grad/Shape_25train/gradients/cross_entropy_total/Mean_grad/Const_1"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1
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
5train/gradients/cross_entropy_total/Mean_grad/truedivRealDiv2train/gradients/cross_entropy_total/Mean_grad/Tile2train/gradients/cross_entropy_total/Mean_grad/Cast"/device:GPU:0*#
_output_shapes
:���������*
T0
�
8train/gradients/cross_entropy_total/Reshape_2_grad/ShapeShape1cross_entropy_total/SoftmaxCrossEntropyWithLogits)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
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
Utrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB :
���������*
dtype0
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
6train/gradients/cross_entropy_total/Reshape_grad/ShapeShapeReadout/predicted_labels)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
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
3train/gradients/Readout/predicted_labels_grad/ShapeShapeReadout/MatMul)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
5train/gradients/Readout/predicted_labels_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:*
dtype0
�
Ctrain/gradients/Readout/predicted_labels_grad/BroadcastGradientArgsBroadcastGradientArgs3train/gradients/Readout/predicted_labels_grad/Shape5train/gradients/Readout/predicted_labels_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
1train/gradients/Readout/predicted_labels_grad/SumSum8train/gradients/cross_entropy_total/Reshape_grad/ReshapeCtrain/gradients/Readout/predicted_labels_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
5train/gradients/Readout/predicted_labels_grad/ReshapeReshape1train/gradients/Readout/predicted_labels_grad/Sum3train/gradients/Readout/predicted_labels_grad/Shape"/device:GPU:0*'
_output_shapes
:���������*
T0*
Tshape0
�
3train/gradients/Readout/predicted_labels_grad/Sum_1Sum8train/gradients/cross_entropy_total/Reshape_grad/ReshapeEtrain/gradients/Readout/predicted_labels_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
7train/gradients/Readout/predicted_labels_grad/Reshape_1Reshape3train/gradients/Readout/predicted_labels_grad/Sum_15train/gradients/Readout/predicted_labels_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:
�
>train/gradients/Readout/predicted_labels_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_16^train/gradients/Readout/predicted_labels_grad/Reshape8^train/gradients/Readout/predicted_labels_grad/Reshape_1"/device:GPU:0
�
Ftrain/gradients/Readout/predicted_labels_grad/tuple/control_dependencyIdentity5train/gradients/Readout/predicted_labels_grad/Reshape?^train/gradients/Readout/predicted_labels_grad/tuple/group_deps"/device:GPU:0*
T0*H
_class>
<:loc:@train/gradients/Readout/predicted_labels_grad/Reshape*'
_output_shapes
:���������
�
Htrain/gradients/Readout/predicted_labels_grad/tuple/control_dependency_1Identity7train/gradients/Readout/predicted_labels_grad/Reshape_1?^train/gradients/Readout/predicted_labels_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:*
T0*J
_class@
><loc:@train/gradients/Readout/predicted_labels_grad/Reshape_1
�
*train/gradients/Readout/MatMul_grad/MatMulMatMulFtrain/gradients/Readout/predicted_labels_grad/tuple/control_dependencyReadout/Variable/read"/device:GPU:0*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
,train/gradients/Readout/MatMul_grad/MatMul_1MatMul'FC1/batch_normalization/batchnorm/add_1Ftrain/gradients/Readout/predicted_labels_grad/tuple/control_dependency"/device:GPU:0*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
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
Btrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ShapeShape'FC1/batch_normalization/batchnorm/mul_1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
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
@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/SumSum<train/gradients/Readout/MatMul_grad/tuple/control_dependencyRtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ReshapeReshape@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/SumBtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape"/device:GPU:0*(
_output_shapes
:����������*
T0*
Tshape0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Sum_1Sum<train/gradients/Readout/MatMul_grad/tuple/control_dependencyTtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Ftrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeBtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Sum_1Dtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
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
Rtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsBtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ShapeDtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mulMulUtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency%FC1/batch_normalization/batchnorm/mul"/device:GPU:0*
T0*(
_output_shapes
:����������
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/SumSum@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mulRtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ReshapeReshape@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/SumBtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Shape"/device:GPU:0*(
_output_shapes
:����������*
T0*
Tshape0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mul_1MulFC1/ReluUtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency"/device:GPU:0*
T0*(
_output_shapes
:����������
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
Utrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityDtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ReshapeN^train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps"/device:GPU:0*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape
�
Wtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityFtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1N^train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps"/device:GPU:0*
_output_shapes	
:�*
T0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1
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
>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/SumSumWtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Ptrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/ReshapeReshape>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Sum@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Shape"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Sum_1SumWtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Rtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/NegNeg@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Sum_1"/device:GPU:0*
_output_shapes
:*
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1Reshape>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/NegBtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Shape_1"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
Ktrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/ReshapeE^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1"/device:GPU:0
�
Strain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityBtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/ReshapeL^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/group_deps"/device:GPU:0*
_output_shapes	
:�*
T0*U
_classK
IGloc:@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape
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
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
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
@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/SumSum@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/mulRtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/BroadcastGradientArgs"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
Utrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityDtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ReshapeN^train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps"/device:GPU:0*
_output_shapes	
:�*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape
�
Wtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityFtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape_1N^train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps"/device:GPU:0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape_1*
_output_shapes	
:�*
T0
�
:train/gradients/FC1/batch_normalization/Squeeze_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0
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
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:�*
dtype0
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
>train/gradients/FC1/batch_normalization/batchnorm/mul_grad/SumSum>train/gradients/FC1/batch_normalization/batchnorm/mul_grad/mulPtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgs"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/ReshapeReshape>train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Sum@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Shape"/device:GPU:0*
Tshape0*
_output_shapes	
:�*
T0
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
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1Reshape@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Sum_1Btrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Shape_1"/device:GPU:0*
Tshape0*
_output_shapes	
:�*
T0
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
Utrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityDtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1L^train/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/group_deps"/device:GPU:0*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1*
_output_shapes	
:�
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
<train/gradients/FC1/batch_normalization/Select_grad/Select_1SelectFC1/batch_normalization/Reshape>train/gradients/FC1/batch_normalization/Select_grad/zeros_like<train/gradients/FC1/batch_normalization/Squeeze_grad/Reshape"/device:GPU:0*
_output_shapes
:	�*
T0
�
Dtrain/gradients/FC1/batch_normalization/Select_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1;^train/gradients/FC1/batch_normalization/Select_grad/Select=^train/gradients/FC1/batch_normalization/Select_grad/Select_1"/device:GPU:0
�
Ltrain/gradients/FC1/batch_normalization/Select_grad/tuple/control_dependencyIdentity:train/gradients/FC1/batch_normalization/Select_grad/SelectE^train/gradients/FC1/batch_normalization/Select_grad/tuple/group_deps"/device:GPU:0*M
_classC
A?loc:@train/gradients/FC1/batch_normalization/Select_grad/Select*
_output_shapes
:	�*
T0
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
?train/gradients/FC1/batch_normalization/ExpandDims_grad/ReshapeReshapeLtrain/gradients/FC1/batch_normalization/Select_grad/tuple/control_dependency=train/gradients/FC1/batch_normalization/ExpandDims_grad/Shape"/device:GPU:0*
Tshape0*
_output_shapes	
:�*
T0
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
Btrain/gradients/FC1/batch_normalization/batchnorm/add_grad/ReshapeReshape>train/gradients/FC1/batch_normalization/batchnorm/add_grad/Sum@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Sum_1SumFtrain/gradients/FC1/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGradRtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Reshape_1Reshape@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Sum_1Btrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes
: 
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
Dtrain/gradients/FC1/batch_normalization/moments/Squeeze_grad/ReshapeReshape?train/gradients/FC1/batch_normalization/ExpandDims_grad/ReshapeBtrain/gradients/FC1/batch_normalization/moments/Squeeze_grad/Shape"/device:GPU:0*
_output_shapes
:	�*
T0*
Tshape0
�
<train/gradients/FC1/batch_normalization/Squeeze_1_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
�
>train/gradients/FC1/batch_normalization/Squeeze_1_grad/ReshapeReshapeStrain/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/control_dependency<train/gradients/FC1/batch_normalization/Squeeze_1_grad/Shape"/device:GPU:0*
Tshape0*
_output_shapes
:	�*
T0
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
Ctrain/gradients/FC1/batch_normalization/moments/variance_grad/ShapeShape1FC1/batch_normalization/moments/SquaredDifference)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/SizeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Atrain/gradients/FC1/batch_normalization/moments/variance_grad/addAdd:FC1/batch_normalization/moments/variance/reduction_indicesBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Size"/device:GPU:0*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
_output_shapes
:
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
Itrain/gradients/FC1/batch_normalization/moments/variance_grad/range/startConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
value	B : *V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0
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
Ktrain/gradients/FC1/batch_normalization/moments/variance_grad/DynamicStitchDynamicStitchCtrain/gradients/FC1/batch_normalization/moments/variance_grad/rangeAtrain/gradients/FC1/batch_normalization/moments/variance_grad/modCtrain/gradients/FC1/batch_normalization/moments/variance_grad/ShapeBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Fill"/device:GPU:0*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
N*#
_output_shapes
:���������
�
Gtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0*
_output_shapes
: 
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/MaximumMaximumKtrain/gradients/FC1/batch_normalization/moments/variance_grad/DynamicStitchGtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum/y"/device:GPU:0*#
_output_shapes
:���������*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape
�
Ftrain/gradients/FC1/batch_normalization/moments/variance_grad/floordivFloorDivCtrain/gradients/FC1/batch_normalization/moments/variance_grad/ShapeEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum"/device:GPU:0*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
_output_shapes
:
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
Dtrain/gradients/FC1/batch_normalization/moments/variance_grad/Prod_1ProdEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_3Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Const_1"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2
�
Itrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
: 
�
Gtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1MaximumDtrain/gradients/FC1/batch_normalization/moments/variance_grad/Prod_1Itrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1/y"/device:GPU:0*
T0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: 
�
Htrain/gradients/FC1/batch_normalization/moments/variance_grad/floordiv_1FloorDivBtrain/gradients/FC1/batch_normalization/moments/variance_grad/ProdGtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1"/device:GPU:0*
T0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: 
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/CastCastHtrain/gradients/FC1/batch_normalization/moments/variance_grad/floordiv_1"/device:GPU:0*
_output_shapes
: *

DstT0*

SrcT0
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/truedivRealDivBtrain/gradients/FC1/batch_normalization/moments/variance_grad/TileBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Cast"/device:GPU:0*
T0*(
_output_shapes
:����������
�
Ltrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ShapeShapeFC1/Relu)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
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
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mulMulMtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/scalarEtrain/gradients/FC1/batch_normalization/moments/variance_grad/truediv"/device:GPU:0*(
_output_shapes
:����������*
T0
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/subSubFC1/Relu,FC1/batch_normalization/moments/StopGradient)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1F^train/gradients/FC1/batch_normalization/moments/variance_grad/truediv"/device:GPU:0*(
_output_shapes
:����������*
T0
�
Ltrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mul_1MulJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mulJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/sub"/device:GPU:0*(
_output_shapes
:����������*
T0
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/SumSumLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mul_1\train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Ntrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ReshapeReshapeJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/SumLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Shape"/device:GPU:0*
Tshape0*(
_output_shapes
:����������*
T0
�
Ltrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Sum_1SumLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mul_1^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Ptrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Reshape_1ReshapeLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Sum_1Ntrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Shape_1"/device:GPU:0*
Tshape0*
_output_shapes
:	�*
T0
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/NegNegPtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Reshape_1"/device:GPU:0*
T0*
_output_shapes
:	�
�
Wtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1O^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ReshapeK^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Neg"/device:GPU:0
�
_train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependencyIdentityNtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ReshapeX^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/group_deps"/device:GPU:0*(
_output_shapes
:����������*
T0*a
_classW
USloc:@train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Reshape
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
>train/gradients/FC1/batch_normalization/moments/mean_grad/SizeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
value	B :*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0
�
=train/gradients/FC1/batch_normalization/moments/mean_grad/addAdd6FC1/batch_normalization/moments/mean/reduction_indices>train/gradients/FC1/batch_normalization/moments/mean_grad/Size"/device:GPU:0*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
_output_shapes
:
�
=train/gradients/FC1/batch_normalization/moments/mean_grad/modFloorMod=train/gradients/FC1/batch_normalization/moments/mean_grad/add>train/gradients/FC1/batch_normalization/moments/mean_grad/Size"/device:GPU:0*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
_output_shapes
:
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
Dtrain/gradients/FC1/batch_normalization/moments/mean_grad/Fill/valueConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/FillFillAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_1Dtrain/gradients/FC1/batch_normalization/moments/mean_grad/Fill/value"/device:GPU:0*
_output_shapes
:*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape
�
Gtrain/gradients/FC1/batch_normalization/moments/mean_grad/DynamicStitchDynamicStitch?train/gradients/FC1/batch_normalization/moments/mean_grad/range=train/gradients/FC1/batch_normalization/moments/mean_grad/mod?train/gradients/FC1/batch_normalization/moments/mean_grad/Shape>train/gradients/FC1/batch_normalization/moments/mean_grad/Fill"/device:GPU:0*#
_output_shapes
:���������*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
N
�
Ctrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
dtype0*
_output_shapes
: 
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/MaximumMaximumGtrain/gradients/FC1/batch_normalization/moments/mean_grad/DynamicStitchCtrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum/y"/device:GPU:0*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*#
_output_shapes
:���������
�
Btrain/gradients/FC1/batch_normalization/moments/mean_grad/floordivFloorDiv?train/gradients/FC1/batch_normalization/moments/mean_grad/ShapeAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum"/device:GPU:0*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
_output_shapes
:
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/ReshapeReshapeDtrain/gradients/FC1/batch_normalization/moments/Squeeze_grad/ReshapeGtrain/gradients/FC1/batch_normalization/moments/mean_grad/DynamicStitch"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/TileTileAtrain/gradients/FC1/batch_normalization/moments/mean_grad/ReshapeBtrain/gradients/FC1/batch_normalization/moments/mean_grad/floordiv"/device:GPU:0*0
_output_shapes
:������������������*

Tmultiples0*
T0
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
?train/gradients/FC1/batch_normalization/moments/mean_grad/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB: *T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
dtype0
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
@train/gradients/FC1/batch_normalization/moments/mean_grad/Prod_1ProdAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_3Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Const_1"/device:GPU:0*
T0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Etrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum_1/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
value	B :*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
dtype0
�
Ctrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum_1Maximum@train/gradients/FC1/batch_normalization/moments/mean_grad/Prod_1Etrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum_1/y"/device:GPU:0*
T0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
_output_shapes
: 
�
Dtrain/gradients/FC1/batch_normalization/moments/mean_grad/floordiv_1FloorDiv>train/gradients/FC1/batch_normalization/moments/mean_grad/ProdCtrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum_1"/device:GPU:0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
_output_shapes
: *
T0
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
 train/gradients/FC1/add_grad/SumSum&train/gradients/FC1/Relu_grad/ReluGrad2train/gradients/FC1/add_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
5train/gradients/FC1/add_grad/tuple/control_dependencyIdentity$train/gradients/FC1/add_grad/Reshape.^train/gradients/FC1/add_grad/tuple/group_deps"/device:GPU:0*7
_class-
+)loc:@train/gradients/FC1/add_grad/Reshape*(
_output_shapes
:����������*
T0
�
7train/gradients/FC1/add_grad/tuple/control_dependency_1Identity&train/gradients/FC1/add_grad/Reshape_1.^train/gradients/FC1/add_grad/tuple/group_deps"/device:GPU:0*
_output_shapes	
:�*
T0*9
_class/
-+loc:@train/gradients/FC1/add_grad/Reshape_1
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
8train/gradients/FC1/MatMul_grad/tuple/control_dependencyIdentity&train/gradients/FC1/MatMul_grad/MatMul1^train/gradients/FC1/MatMul_grad/tuple/group_deps"/device:GPU:0*)
_output_shapes
:�����������*
T0*9
_class/
-+loc:@train/gradients/FC1/MatMul_grad/MatMul
�
:train/gradients/FC1/MatMul_grad/tuple/control_dependency_1Identity(train/gradients/FC1/MatMul_grad/MatMul_11^train/gradients/FC1/MatMul_grad/tuple/group_deps"/device:GPU:0*;
_class1
/-loc:@train/gradients/FC1/MatMul_grad/MatMul_1*!
_output_shapes
:���*
T0
�
&train/gradients/FC1/Reshape_grad/ShapeShape#CNN3/batch_normalization/cond/Merge)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
(train/gradients/FC1/Reshape_grad/ReshapeReshape8train/gradients/FC1/MatMul_grad/tuple/control_dependency&train/gradients/FC1/Reshape_grad/Shape"/device:GPU:0*/
_output_shapes
:���������@*
T0*
Tshape0
�
Btrain/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_gradSwitch(train/gradients/FC1/Reshape_grad/Reshape%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*;
_class1
/-loc:@train/gradients/FC1/Reshape_grad/Reshape*J
_output_shapes8
6:���������@:���������@*
T0
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
Rtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*%
valueB"             *
dtype0
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
Otrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1	TransposeQtrain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/control_dependencyTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/perm"/device:GPU:0*
T0*/
_output_shapes
:���������@*
Tperm0
�
Vtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradOtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1Mtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
data_formatNHWC*G
_output_shapes5
3:���������@::::*
is_training( *
epsilon%o�:*
T0
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
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityOtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*b
_classX
VTloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2*/
_output_shapes
:���������@*
T0
�
^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityXtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityXtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
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
train/gradients/zeros_like_8	ZerosLike.CNN3/batch_normalization/cond/FusedBatchNorm:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:
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
Ztrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradS^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������@*
T0
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
: *
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
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
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMerge\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencytrain/gradients/zeros"/device:GPU:0*1
_output_shapes
:���������@: *
T0*
N
�
train/gradients/Switch_1Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
::*
T0
�
train/gradients/Shape_2Shapetrain/gradients/Switch_1:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_1/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
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
train/gradients/Switch_2Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
::*
T0
�
train/gradients/Shape_3Shapetrain/gradients/Switch_2:1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
train/gradients/zeros_2/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_2Filltrain/gradients/Shape_3train/gradients/zeros_2/Const"/device:GPU:0*
T0*
_output_shapes
:
�
Vtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMerge^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2train/gradients/zeros_2"/device:GPU:0*
N*
_output_shapes

:: *
T0
�
train/gradients/Switch_3Switch	CNN3/Relu%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*J
_output_shapes8
6:���������@:���������@
~
train/gradients/Shape_4Shapetrain/gradients/Switch_3"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
train/gradients/zeros_3/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_3Filltrain/gradients/Shape_4train/gradients/zeros_3/Const"/device:GPU:0*
T0*/
_output_shapes
:���������@
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
train/gradients/zeros_4Filltrain/gradients/Shape_5train/gradients/zeros_4/Const"/device:GPU:0*
_output_shapes
:*
T0
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMerge\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1train/gradients/zeros_4"/device:GPU:0*
N*
_output_shapes

:: *
T0
�
train/gradients/Switch_5Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
::*
T0
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
train/gradients/zeros_5Filltrain/gradients/Shape_6train/gradients/zeros_5/Const"/device:GPU:0*
_output_shapes
:*
T0
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMerge\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2train/gradients/zeros_5"/device:GPU:0*
_output_shapes

:: *
T0*
N
�
train/gradients/AddN_2AddNTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradRtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������@
�
'train/gradients/CNN3/Relu_grad/ReluGradReluGradtrain/gradients/AddN_2	CNN3/Relu"/device:GPU:0*
T0*/
_output_shapes
:���������@
�
train/gradients/AddN_3AddNVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
train/gradients/AddN_4AddNVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:
�
#train/gradients/CNN3/add_grad/ShapeShapeCNN3/Conv2D)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
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
!train/gradients/CNN3/add_grad/SumSum'train/gradients/CNN3/Relu_grad/ReluGrad3train/gradients/CNN3/add_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
%train/gradients/CNN3/add_grad/ReshapeReshape!train/gradients/CNN3/add_grad/Sum#train/gradients/CNN3/add_grad/Shape"/device:GPU:0*
Tshape0*/
_output_shapes
:���������@*
T0
�
#train/gradients/CNN3/add_grad/Sum_1Sum'train/gradients/CNN3/Relu_grad/ReluGrad5train/gradients/CNN3/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
'train/gradients/CNN3/add_grad/Reshape_1Reshape#train/gradients/CNN3/add_grad/Sum_1%train/gradients/CNN3/add_grad/Shape_1"/device:GPU:0*
_output_shapes
:@*
T0*
Tshape0
�
.train/gradients/CNN3/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1&^train/gradients/CNN3/add_grad/Reshape(^train/gradients/CNN3/add_grad/Reshape_1"/device:GPU:0
�
6train/gradients/CNN3/add_grad/tuple/control_dependencyIdentity%train/gradients/CNN3/add_grad/Reshape/^train/gradients/CNN3/add_grad/tuple/group_deps"/device:GPU:0*
T0*8
_class.
,*loc:@train/gradients/CNN3/add_grad/Reshape*/
_output_shapes
:���������@
�
8train/gradients/CNN3/add_grad/tuple/control_dependency_1Identity'train/gradients/CNN3/add_grad/Reshape_1/^train/gradients/CNN3/add_grad/tuple/group_deps"/device:GPU:0*
T0*:
_class0
.,loc:@train/gradients/CNN3/add_grad/Reshape_1*
_output_shapes
:@
�
'train/gradients/CNN3/Conv2D_grad/ShapeNShapeN#CNN2/batch_normalization/cond/MergeCNN3/weights/Variable/read)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
N* 
_output_shapes
::
�
4train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'train/gradients/CNN3/Conv2D_grad/ShapeNCNN3/weights/Variable/read6train/gradients/CNN3/add_grad/tuple/control_dependency"/device:GPU:0*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
5train/gradients/CNN3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter#CNN2/batch_normalization/cond/Merge)train/gradients/CNN3/Conv2D_grad/ShapeN:16train/gradients/CNN3/add_grad/tuple/control_dependency"/device:GPU:0*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
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
;train/gradients/CNN3/Conv2D_grad/tuple/control_dependency_1Identity5train/gradients/CNN3/Conv2D_grad/Conv2DBackpropFilter2^train/gradients/CNN3/Conv2D_grad/tuple/group_deps"/device:GPU:0*
T0*H
_class>
<:loc:@train/gradients/CNN3/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @
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
train/gradients/zeros_like_10	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
train/gradients/zeros_like_11	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
T0
�
train/gradients/zeros_like_12	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
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
Otrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2	TransposeVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/perm"/device:GPU:0*/
_output_shapes
:���������   *
Tperm0*
T0
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
^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityXtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityXtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
: 
�
train/gradients/zeros_like_13	ZerosLike.CNN2/batch_normalization/cond/FusedBatchNorm:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
: 
�
train/gradients/zeros_like_14	ZerosLike.CNN2/batch_normalization/cond/FusedBatchNorm:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
T0
�
train/gradients/zeros_like_15	ZerosLike.CNN2/batch_normalization/cond/FusedBatchNorm:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
T0
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
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
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
train/gradients/Switch_6Switch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*J
_output_shapes8
6:���������   :���������   
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
train/gradients/zeros_6Filltrain/gradients/Shape_7train/gradients/zeros_6/Const"/device:GPU:0*
T0*/
_output_shapes
:���������   
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMerge\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencytrain/gradients/zeros_6"/device:GPU:0*
N*1
_output_shapes
:���������   : *
T0
�
train/gradients/Switch_7Switch batch_normalization_1/gamma/read%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
: : *
T0
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
train/gradients/Switch_8Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
: : *
T0
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
train/gradients/zeros_8Filltrain/gradients/Shape_9train/gradients/zeros_8/Const"/device:GPU:0*
T0*
_output_shapes
: 
�
Vtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradMerge^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2train/gradients/zeros_8"/device:GPU:0*
_output_shapes

: : *
T0*
N
�
train/gradients/Switch_9Switch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*J
_output_shapes8
6:���������   :���������   

train/gradients/Shape_10Shapetrain/gradients/Switch_9"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
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
train/gradients/Shape_12Shapetrain/gradients/Switch_11"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
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
train/gradients/AddN_5AddNTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradRtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_grad"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_grad*
N*/
_output_shapes
:���������   *
T0
�
'train/gradients/CNN2/Relu_grad/ReluGradReluGradtrain/gradients/AddN_5	CNN2/Relu"/device:GPU:0*
T0*/
_output_shapes
:���������   
�
train/gradients/AddN_6AddNVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad"/device:GPU:0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
: *
T0
�
train/gradients/AddN_7AddNVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad"/device:GPU:0*
_output_shapes
: *
T0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N
�
#train/gradients/CNN2/add_grad/ShapeShapeCNN2/Conv2D)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
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
!train/gradients/CNN2/add_grad/SumSum'train/gradients/CNN2/Relu_grad/ReluGrad3train/gradients/CNN2/add_grad/BroadcastGradientArgs"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
6train/gradients/CNN2/add_grad/tuple/control_dependencyIdentity%train/gradients/CNN2/add_grad/Reshape/^train/gradients/CNN2/add_grad/tuple/group_deps"/device:GPU:0*
T0*8
_class.
,*loc:@train/gradients/CNN2/add_grad/Reshape*/
_output_shapes
:���������   
�
8train/gradients/CNN2/add_grad/tuple/control_dependency_1Identity'train/gradients/CNN2/add_grad/Reshape_1/^train/gradients/CNN2/add_grad/tuple/group_deps"/device:GPU:0*:
_class0
.,loc:@train/gradients/CNN2/add_grad/Reshape_1*
_output_shapes
: *
T0
�
'train/gradients/CNN2/Conv2D_grad/ShapeNShapeN#CNN1/batch_normalization/cond/MergeCNN2/weights/Variable/read)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
N* 
_output_shapes
::*
T0
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
;train/gradients/CNN2/Conv2D_grad/tuple/control_dependency_1Identity5train/gradients/CNN2/Conv2D_grad/Conv2DBackpropFilter2^train/gradients/CNN2/Conv2D_grad/tuple/group_deps"/device:GPU:0*
T0*H
_class>
<:loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
�
Btrain/gradients/CNN1/batch_normalization/cond/Merge_grad/cond_gradSwitch9train/gradients/CNN2/Conv2D_grad/tuple/control_dependency%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*G
_class=
;9loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput*J
_output_shapes8
6:���������@@:���������@@*
T0
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
Strain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/control_dependency_1IdentityDtrain/gradients/CNN1/batch_normalization/cond/Merge_grad/cond_grad:1J^train/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*
T0*G
_class=
;9loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������@@
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
Rtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Mtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose	Transpose5CNN1/batch_normalization/cond/FusedBatchNorm_1/SwitchRtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/perm"/device:GPU:0*/
_output_shapes
:���������@@*
Tperm0*
T0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
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
Otrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2	TransposeVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/perm"/device:GPU:0*
T0*/
_output_shapes
:���������@@*
Tperm0
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
train/gradients/zeros_like_21	ZerosLike.CNN1/batch_normalization/cond/FusedBatchNorm:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:@
�
train/gradients/zeros_like_22	ZerosLike.CNN1/batch_normalization/cond/FusedBatchNorm:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:@
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
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@*
T0
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@*
T0
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
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
train/gradients/zeros_12/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_12Filltrain/gradients/Shape_13train/gradients/zeros_12/Const"/device:GPU:0*
T0*/
_output_shapes
:���������@@
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_grad/cond_gradMerge\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencytrain/gradients/zeros_12"/device:GPU:0*
T0*
N*1
_output_shapes
:���������@@: 
�
train/gradients/Switch_13Switchbatch_normalization/gamma/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
:@:@
�
train/gradients/Shape_14Shapetrain/gradients/Switch_13:1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
train/gradients/zeros_13/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_13Filltrain/gradients/Shape_14train/gradients/zeros_13/Const"/device:GPU:0*
_output_shapes
:@*
T0
�
Vtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMerge^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1train/gradients/zeros_13"/device:GPU:0*
T0*
N*
_output_shapes

:@: 
�
train/gradients/Switch_14Switchbatch_normalization/beta/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
:@:@
�
train/gradients/Shape_15Shapetrain/gradients/Switch_14:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_14/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
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
train/gradients/Shape_16Shapetrain/gradients/Switch_15"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
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
train/gradients/zeros_16/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
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
train/gradients/zeros_17Filltrain/gradients/Shape_18train/gradients/zeros_17/Const"/device:GPU:0*
T0*
_output_shapes
:@
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
train/gradients/AddN_10AddNVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N*
_output_shapes
:@
�
#train/gradients/CNN1/add_grad/ShapeShapeCNN1/Conv2D)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
%train/gradients/CNN1/add_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
3train/gradients/CNN1/add_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/CNN1/add_grad/Shape%train/gradients/CNN1/add_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
!train/gradients/CNN1/add_grad/SumSum'train/gradients/CNN1/Relu_grad/ReluGrad3train/gradients/CNN1/add_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
%train/gradients/CNN1/add_grad/ReshapeReshape!train/gradients/CNN1/add_grad/Sum#train/gradients/CNN1/add_grad/Shape"/device:GPU:0*
Tshape0*/
_output_shapes
:���������@@*
T0
�
#train/gradients/CNN1/add_grad/Sum_1Sum'train/gradients/CNN1/Relu_grad/ReluGrad5train/gradients/CNN1/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
'train/gradients/CNN1/add_grad/Reshape_1Reshape#train/gradients/CNN1/add_grad/Sum_1%train/gradients/CNN1/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:
�
.train/gradients/CNN1/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1&^train/gradients/CNN1/add_grad/Reshape(^train/gradients/CNN1/add_grad/Reshape_1"/device:GPU:0
�
6train/gradients/CNN1/add_grad/tuple/control_dependencyIdentity%train/gradients/CNN1/add_grad/Reshape/^train/gradients/CNN1/add_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������@@*
T0*8
_class.
,*loc:@train/gradients/CNN1/add_grad/Reshape
�
8train/gradients/CNN1/add_grad/tuple/control_dependency_1Identity'train/gradients/CNN1/add_grad/Reshape_1/^train/gradients/CNN1/add_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:*
T0*:
_class0
.,loc:@train/gradients/CNN1/add_grad/Reshape_1
�
'train/gradients/CNN1/Conv2D_grad/ShapeNShapeNinput/imagesCNN1/weights/Variable/read)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
::*
T0*
out_type0*
N
�
4train/gradients/CNN1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'train/gradients/CNN1/Conv2D_grad/ShapeNCNN1/weights/Variable/read6train/gradients/CNN1/add_grad/tuple/control_dependency"/device:GPU:0*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
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
9train/gradients/CNN1/Conv2D_grad/tuple/control_dependencyIdentity4train/gradients/CNN1/Conv2D_grad/Conv2DBackpropInput2^train/gradients/CNN1/Conv2D_grad/tuple/group_deps"/device:GPU:0*G
_class=
;9loc:@train/gradients/CNN1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:���������@@*
T0
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
train/beta2_power/initial_valueConst"/device:GPU:0*
valueB
 *w�?*'
_class
loc:@CNN1/biases/Variable*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
train/beta2_power/readIdentitytrain/beta2_power"/device:GPU:0*
T0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
: 
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
!CNN1/weights/Variable/Adam/AssignAssignCNN1/weights/Variable/Adam,CNN1/weights/Variable/Adam/Initializer/zeros"/device:GPU:0*&
_output_shapes
:*
use_locking(*
T0*(
_class
loc:@CNN1/weights/Variable*
validate_shape(
�
CNN1/weights/Variable/Adam/readIdentityCNN1/weights/Variable/Adam"/device:GPU:0*
T0*(
_class
loc:@CNN1/weights/Variable*&
_output_shapes
:
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
VariableV2"/device:GPU:0*
_output_shapes
:*
shared_name *'
_class
loc:@CNN1/biases/Variable*
	container *
shape:*
dtype0
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
%batch_normalization/gamma/Adam/AssignAssignbatch_normalization/gamma/Adam0batch_normalization/gamma/Adam/Initializer/zeros"/device:GPU:0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
#batch_normalization/gamma/Adam/readIdentitybatch_normalization/gamma/Adam"/device:GPU:0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@*
T0
�
2batch_normalization/gamma/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:@*,
_class"
 loc:@batch_normalization/gamma*
valueB@*    *
dtype0
�
 batch_normalization/gamma/Adam_1
VariableV2"/device:GPU:0*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *,
_class"
 loc:@batch_normalization/gamma
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
%batch_normalization/gamma/Adam_1/readIdentity batch_normalization/gamma/Adam_1"/device:GPU:0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@*
T0
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
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container *
shape:@
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
VariableV2"/device:GPU:0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container *
shape:@*
dtype0
�
&batch_normalization/beta/Adam_1/AssignAssignbatch_normalization/beta/Adam_11batch_normalization/beta/Adam_1/Initializer/zeros"/device:GPU:0*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
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
VariableV2"/device:GPU:0*(
_class
loc:@CNN2/weights/Variable*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name 
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
CNN2/weights/Variable/Adam/readIdentityCNN2/weights/Variable/Adam"/device:GPU:0*&
_output_shapes
: *
T0*(
_class
loc:@CNN2/weights/Variable
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
VariableV2"/device:GPU:0*(
_class
loc:@CNN2/weights/Variable*
	container *
shape: *
dtype0*&
_output_shapes
: *
shared_name 
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
!CNN2/weights/Variable/Adam_1/readIdentityCNN2/weights/Variable/Adam_1"/device:GPU:0*(
_class
loc:@CNN2/weights/Variable*&
_output_shapes
: *
T0
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
VariableV2"/device:GPU:0*
_output_shapes
: *
shared_name *'
_class
loc:@CNN2/biases/Variable*
	container *
shape: *
dtype0
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
CNN2/biases/Variable/Adam/readIdentityCNN2/biases/Variable/Adam"/device:GPU:0*
_output_shapes
: *
T0*'
_class
loc:@CNN2/biases/Variable
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
VariableV2"/device:GPU:0*'
_class
loc:@CNN2/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
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
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@batch_normalization_1/gamma
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
%batch_normalization_1/gamma/Adam/readIdentity batch_normalization_1/gamma/Adam"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: *
T0
�
4batch_normalization_1/gamma/Adam_1/Initializer/zerosConst"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma*
valueB *    *
dtype0*
_output_shapes
: 
�
"batch_normalization_1/gamma/Adam_1
VariableV2"/device:GPU:0*
_output_shapes
: *
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape: *
dtype0
�
)batch_normalization_1/gamma/Adam_1/AssignAssign"batch_normalization_1/gamma/Adam_14batch_normalization_1/gamma/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(*
_output_shapes
: 
�
'batch_normalization_1/gamma/Adam_1/readIdentity"batch_normalization_1/gamma/Adam_1"/device:GPU:0*
_output_shapes
: *
T0*.
_class$
" loc:@batch_normalization_1/gamma
�
1batch_normalization_1/beta/Adam/Initializer/zerosConst"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta*
valueB *    *
dtype0*
_output_shapes
: 
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
&batch_normalization_1/beta/Adam/AssignAssignbatch_normalization_1/beta/Adam1batch_normalization_1/beta/Adam/Initializer/zeros"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
VariableV2"/device:GPU:0*
shape: *
dtype0*
_output_shapes
: *
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container 
�
(batch_normalization_1/beta/Adam_1/AssignAssign!batch_normalization_1/beta/Adam_13batch_normalization_1/beta/Adam_1/Initializer/zeros"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
&batch_normalization_1/beta/Adam_1/readIdentity!batch_normalization_1/beta/Adam_1"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: 
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
VariableV2"/device:GPU:0*
shared_name *(
_class
loc:@CNN3/weights/Variable*
	container *
shape: @*
dtype0*&
_output_shapes
: @
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
#CNN3/weights/Variable/Adam_1/AssignAssignCNN3/weights/Variable/Adam_1.CNN3/weights/Variable/Adam_1/Initializer/zeros"/device:GPU:0*&
_output_shapes
: @*
use_locking(*
T0*(
_class
loc:@CNN3/weights/Variable*
validate_shape(
�
!CNN3/weights/Variable/Adam_1/readIdentityCNN3/weights/Variable/Adam_1"/device:GPU:0*&
_output_shapes
: @*
T0*(
_class
loc:@CNN3/weights/Variable
�
+CNN3/biases/Variable/Adam/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:@*'
_class
loc:@CNN3/biases/Variable*
valueB@*    *
dtype0
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
 CNN3/biases/Variable/Adam/AssignAssignCNN3/biases/Variable/Adam+CNN3/biases/Variable/Adam/Initializer/zeros"/device:GPU:0*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@CNN3/biases/Variable*
validate_shape(
�
CNN3/biases/Variable/Adam/readIdentityCNN3/biases/Variable/Adam"/device:GPU:0*
_output_shapes
:@*
T0*'
_class
loc:@CNN3/biases/Variable
�
-CNN3/biases/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
valueB@*    *
dtype0*
_output_shapes
:@
�
CNN3/biases/Variable/Adam_1
VariableV2"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
"CNN3/biases/Variable/Adam_1/AssignAssignCNN3/biases/Variable/Adam_1-CNN3/biases/Variable/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN3/biases/Variable*
validate_shape(*
_output_shapes
:@
�
 CNN3/biases/Variable/Adam_1/readIdentityCNN3/biases/Variable/Adam_1"/device:GPU:0*
T0*'
_class
loc:@CNN3/biases/Variable*
_output_shapes
:@
�
2batch_normalization_2/gamma/Adam/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*    *
dtype0
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
'batch_normalization_2/gamma/Adam/AssignAssign batch_normalization_2/gamma/Adam2batch_normalization_2/gamma/Adam/Initializer/zeros"/device:GPU:0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
%batch_normalization_2/gamma/Adam/readIdentity batch_normalization_2/gamma/Adam"/device:GPU:0*
_output_shapes
:*
T0*.
_class$
" loc:@batch_normalization_2/gamma
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
)batch_normalization_2/gamma/Adam_1/AssignAssign"batch_normalization_2/gamma/Adam_14batch_normalization_2/gamma/Adam_1/Initializer/zeros"/device:GPU:0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
'batch_normalization_2/gamma/Adam_1/readIdentity"batch_normalization_2/gamma/Adam_1"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:
�
1batch_normalization_2/beta/Adam/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
dtype0
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
#FC1/Variable/Adam/Initializer/zerosConst"/device:GPU:0*
_class
loc:@FC1/Variable* 
valueB���*    *
dtype0*!
_output_shapes
:���
�
FC1/Variable/Adam
VariableV2"/device:GPU:0*
	container *
shape:���*
dtype0*!
_output_shapes
:���*
shared_name *
_class
loc:@FC1/Variable
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
FC1/Variable/Adam/readIdentityFC1/Variable/Adam"/device:GPU:0*
T0*
_class
loc:@FC1/Variable*!
_output_shapes
:���
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
FC1/Variable/Adam_1/AssignAssignFC1/Variable/Adam_1%FC1/Variable/Adam_1/Initializer/zeros"/device:GPU:0*
_class
loc:@FC1/Variable*
validate_shape(*!
_output_shapes
:���*
use_locking(*
T0
�
FC1/Variable/Adam_1/readIdentityFC1/Variable/Adam_1"/device:GPU:0*
T0*
_class
loc:@FC1/Variable*!
_output_shapes
:���
�
%FC1/Variable_1/Adam/Initializer/zerosConst"/device:GPU:0*
_output_shapes	
:�*!
_class
loc:@FC1/Variable_1*
valueB�*    *
dtype0
�
FC1/Variable_1/Adam
VariableV2"/device:GPU:0*
shared_name *!
_class
loc:@FC1/Variable_1*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
FC1/Variable_1/Adam/AssignAssignFC1/Variable_1/Adam%FC1/Variable_1/Adam/Initializer/zeros"/device:GPU:0*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@FC1/Variable_1*
validate_shape(
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
VariableV2"/device:GPU:0*
shared_name *!
_class
loc:@FC1/Variable_1*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
FC1/Variable_1/Adam_1/AssignAssignFC1/Variable_1/Adam_1'FC1/Variable_1/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@FC1/Variable_1*
validate_shape(
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
VariableV2"/device:GPU:0*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container 
�
)batch_normalization_3/gamma/Adam_1/AssignAssign"batch_normalization_3/gamma/Adam_14batch_normalization_3/gamma/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes	
:�*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
validate_shape(
�
'batch_normalization_3/gamma/Adam_1/readIdentity"batch_normalization_3/gamma/Adam_1"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:�
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
$batch_normalization_3/beta/Adam/readIdentitybatch_normalization_3/beta/Adam"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:�
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
(batch_normalization_3/beta/Adam_1/AssignAssign!batch_normalization_3/beta/Adam_13batch_normalization_3/beta/Adam_1/Initializer/zeros"/device:GPU:0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
&batch_normalization_3/beta/Adam_1/readIdentity!batch_normalization_3/beta/Adam_1"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:�
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
Readout/Variable/Adam/AssignAssignReadout/Variable/Adam'Readout/Variable/Adam/Initializer/zeros"/device:GPU:0*
_output_shapes
:	�*
use_locking(*
T0*#
_class
loc:@Readout/Variable*
validate_shape(
�
Readout/Variable/Adam/readIdentityReadout/Variable/Adam"/device:GPU:0*
T0*#
_class
loc:@Readout/Variable*
_output_shapes
:	�
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
+Readout/Variable_1/Adam_1/Initializer/zerosConst"/device:GPU:0*%
_class
loc:@Readout/Variable_1*
valueB*    *
dtype0*
_output_shapes
:
�
Readout/Variable_1/Adam_1
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
 Readout/Variable_1/Adam_1/AssignAssignReadout/Variable_1/Adam_1+Readout/Variable_1/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@Readout/Variable_1*
validate_shape(
�
Readout/Variable_1/Adam_1/readIdentityReadout/Variable_1/Adam_1"/device:GPU:0*
_output_shapes
:*
T0*%
_class
loc:@Readout/Variable_1
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
train/Adam/epsilonConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
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
0train/Adam/update_CNN1/biases/Variable/ApplyAdam	ApplyAdamCNN1/biases/VariableCNN1/biases/Variable/AdamCNN1/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon8train/gradients/CNN1/add_grad/tuple/control_dependency_1"/device:GPU:0*
_output_shapes
:*
use_locking( *
T0*'
_class
loc:@CNN1/biases/Variable*
use_nesterov( 
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
7train/Adam/update_batch_normalization_1/gamma/ApplyAdam	ApplyAdambatch_normalization_1/gamma batch_normalization_1/gamma/Adam"batch_normalization_1/gamma/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_6"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma*
use_nesterov( *
_output_shapes
: *
use_locking( *
T0
�
6train/Adam/update_batch_normalization_1/beta/ApplyAdam	ApplyAdambatch_normalization_1/betabatch_normalization_1/beta/Adam!batch_normalization_1/beta/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_7"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta*
use_nesterov( *
_output_shapes
: *
use_locking( *
T0
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
(train/Adam/update_FC1/Variable/ApplyAdam	ApplyAdamFC1/VariableFC1/Variable/AdamFC1/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/FC1/MatMul_grad/tuple/control_dependency_1"/device:GPU:0*
use_locking( *
T0*
_class
loc:@FC1/Variable*
use_nesterov( *!
_output_shapes
:���
�
*train/Adam/update_FC1/Variable_1/ApplyAdam	ApplyAdamFC1/Variable_1FC1/Variable_1/AdamFC1/Variable_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon7train/gradients/FC1/add_grad/tuple/control_dependency_1"/device:GPU:0*
_output_shapes	
:�*
use_locking( *
T0*!
_class
loc:@FC1/Variable_1*
use_nesterov( 
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
6train/Adam/update_batch_normalization_3/beta/ApplyAdam	ApplyAdambatch_normalization_3/betabatch_normalization_3/beta/Adam!batch_normalization_3/beta/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonStrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependency"/device:GPU:0*-
_class#
!loc:@batch_normalization_3/beta*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
,train/Adam/update_Readout/Variable/ApplyAdam	ApplyAdamReadout/VariableReadout/Variable/AdamReadout/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/Readout/MatMul_grad/tuple/control_dependency_1"/device:GPU:0*#
_class
loc:@Readout/Variable*
use_nesterov( *
_output_shapes
:	�*
use_locking( *
T0
�
.train/Adam/update_Readout/Variable_1/ApplyAdam	ApplyAdamReadout/Variable_1Readout/Variable_1/AdamReadout/Variable_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonHtrain/gradients/Readout/predicted_labels_grad/tuple/control_dependency_1"/device:GPU:0*
use_locking( *
T0*%
_class
loc:@Readout/Variable_1*
use_nesterov( *
_output_shapes
:
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta12^train/Adam/update_CNN1/weights/Variable/ApplyAdam1^train/Adam/update_CNN1/biases/Variable/ApplyAdam6^train/Adam/update_batch_normalization/gamma/ApplyAdam5^train/Adam/update_batch_normalization/beta/ApplyAdam2^train/Adam/update_CNN2/weights/Variable/ApplyAdam1^train/Adam/update_CNN2/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_1/gamma/ApplyAdam7^train/Adam/update_batch_normalization_1/beta/ApplyAdam2^train/Adam/update_CNN3/weights/Variable/ApplyAdam1^train/Adam/update_CNN3/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_2/gamma/ApplyAdam7^train/Adam/update_batch_normalization_2/beta/ApplyAdam)^train/Adam/update_FC1/Variable/ApplyAdam+^train/Adam/update_FC1/Variable_1/ApplyAdam8^train/Adam/update_batch_normalization_3/gamma/ApplyAdam7^train/Adam/update_batch_normalization_3/beta/ApplyAdam-^train/Adam/update_Readout/Variable/ApplyAdam/^train/Adam/update_Readout/Variable_1/ApplyAdam"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
: *
T0
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
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta22^train/Adam/update_CNN1/weights/Variable/ApplyAdam1^train/Adam/update_CNN1/biases/Variable/ApplyAdam6^train/Adam/update_batch_normalization/gamma/ApplyAdam5^train/Adam/update_batch_normalization/beta/ApplyAdam2^train/Adam/update_CNN2/weights/Variable/ApplyAdam1^train/Adam/update_CNN2/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_1/gamma/ApplyAdam7^train/Adam/update_batch_normalization_1/beta/ApplyAdam2^train/Adam/update_CNN3/weights/Variable/ApplyAdam1^train/Adam/update_CNN3/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_2/gamma/ApplyAdam7^train/Adam/update_batch_normalization_2/beta/ApplyAdam)^train/Adam/update_FC1/Variable/ApplyAdam+^train/Adam/update_FC1/Variable_1/ApplyAdam8^train/Adam/update_batch_normalization_3/gamma/ApplyAdam7^train/Adam/update_batch_normalization_3/beta/ApplyAdam-^train/Adam/update_Readout/Variable/ApplyAdam/^train/Adam/update_Readout/Variable_1/ApplyAdam"/device:GPU:0*
T0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�


train/AdamNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_12^train/Adam/update_CNN1/weights/Variable/ApplyAdam1^train/Adam/update_CNN1/biases/Variable/ApplyAdam6^train/Adam/update_batch_normalization/gamma/ApplyAdam5^train/Adam/update_batch_normalization/beta/ApplyAdam2^train/Adam/update_CNN2/weights/Variable/ApplyAdam1^train/Adam/update_CNN2/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_1/gamma/ApplyAdam7^train/Adam/update_batch_normalization_1/beta/ApplyAdam2^train/Adam/update_CNN3/weights/Variable/ApplyAdam1^train/Adam/update_CNN3/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_2/gamma/ApplyAdam7^train/Adam/update_batch_normalization_2/beta/ApplyAdam)^train/Adam/update_FC1/Variable/ApplyAdam+^train/Adam/update_FC1/Variable_1/ApplyAdam8^train/Adam/update_batch_normalization_3/gamma/ApplyAdam7^train/Adam/update_batch_normalization_3/beta/ApplyAdam-^train/Adam/update_Readout/Variable/ApplyAdam/^train/Adam/update_Readout/Variable_1/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1"/device:GPU:0
s
"accuracy/correct_results/dimensionConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
accuracy/correct_resultsArgMaxinput/correct_labels"accuracy/correct_results/dimension"/device:GPU:0*
output_type0	*#
_output_shapes
:���������*

Tidx0*
T0
�
accuracy/results_matchEqualaccuracy/correct_resultsReadout/predicted_results"/device:GPU:0*
T0	*#
_output_shapes
:���������
y
accuracy/CastCastaccuracy/results_match"/device:GPU:0*#
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
accuracy/accuracyMeanaccuracy/Castaccuracy/Const"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
: *
shared_name *5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
	container *
shape: 
�
)accuracy_streaming_mean/mean/total/AssignAssign"accuracy_streaming_mean/mean/total4accuracy_streaming_mean/mean/total/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
validate_shape(*
_output_shapes
: 
�
'accuracy_streaming_mean/mean/total/readIdentity"accuracy_streaming_mean/mean/total"/device:GPU:0*
_output_shapes
: *
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/total
�
4accuracy_streaming_mean/mean/count/Initializer/zerosConst*
_output_shapes
: *5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
valueB
 *    *
dtype0
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
!accuracy_streaming_mean/mean/SizeConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
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
 accuracy_streaming_mean/mean/SumSumaccuracy/accuracy"accuracy_streaming_mean/mean/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
&accuracy_streaming_mean/mean/AssignAdd	AssignAdd"accuracy_streaming_mean/mean/total accuracy_streaming_mean/mean/Sum"/device:GPU:0*
_output_shapes
: *
use_locking( *
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/total
�
(accuracy_streaming_mean/mean/AssignAdd_1	AssignAdd"accuracy_streaming_mean/mean/count&accuracy_streaming_mean/mean/ToFloat_1^accuracy/accuracy"/device:GPU:0*
_output_shapes
: *
use_locking( *
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/count
z
&accuracy_streaming_mean/mean/Greater/yConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
$accuracy_streaming_mean/mean/GreaterGreater'accuracy_streaming_mean/mean/count/read&accuracy_streaming_mean/mean/Greater/y"/device:GPU:0*
_output_shapes
: *
T0
�
$accuracy_streaming_mean/mean/truedivRealDiv'accuracy_streaming_mean/mean/total/read'accuracy_streaming_mean/mean/count/read"/device:GPU:0*
_output_shapes
: *
T0
x
$accuracy_streaming_mean/mean/value/eConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
"accuracy_streaming_mean/mean/valueSelect$accuracy_streaming_mean/mean/Greater$accuracy_streaming_mean/mean/truediv$accuracy_streaming_mean/mean/value/e"/device:GPU:0*
_output_shapes
: *
T0
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
(accuracy_streaming_mean/mean/update_op/eConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
�
&accuracy_streaming_mean/mean/update_opSelect&accuracy_streaming_mean/mean/Greater_1&accuracy_streaming_mean/mean/truediv_1(accuracy_streaming_mean/mean/update_op/e"/device:GPU:0*
T0*
_output_shapes
: 
�
accuracy_streaming_mean/initNoOp*^accuracy_streaming_mean/mean/total/Assign*^accuracy_streaming_mean/mean/count/Assign"/device:GPU:0
�
accuracy_streaming_mean_1/tagsConst"/device:GPU:0**
value!B Baccuracy_streaming_mean_1*
dtype0*
_output_shapes
: 
�
accuracy_streaming_mean_1ScalarSummaryaccuracy_streaming_mean_1/tags"accuracy_streaming_mean/mean/value"/device:GPU:0*
T0*
_output_shapes
: 
�
Merge/MergeSummaryMergeSummaryCNN1/weights/summaries/meanCNN1/weights/summaries/stddev_1CNN1/weights/summaries/maxCNN1/weights/summaries/min CNN1/weights/summaries/histogramCNN1/biases/summaries/meanCNN1/biases/summaries/stddev_1CNN1/biases/summaries/maxCNN1/biases/summaries/minCNN1/biases/summaries/histogramCNN1/activationsCNN1/batch_normCNN2/weights/summaries/meanCNN2/weights/summaries/stddev_1CNN2/weights/summaries/maxCNN2/weights/summaries/min CNN2/weights/summaries/histogramCNN2/biases/summaries/meanCNN2/biases/summaries/stddev_1CNN2/biases/summaries/maxCNN2/biases/summaries/minCNN2/biases/summaries/histogramCNN2/activationsCNN2/batch_normCNN3/weights/summaries/meanCNN3/weights/summaries/stddev_1CNN3/weights/summaries/maxCNN3/weights/summaries/min CNN3/weights/summaries/histogramCNN3/biases/summaries/meanCNN3/biases/summaries/stddev_1CNN3/biases/summaries/maxCNN3/biases/summaries/minCNN3/biases/summaries/histogramCNN3/activationsCNN3/batch_normaccuracy_streaming_mean_1"/device:GPU:0*
_output_shapes
: *
N%"��d�l\     �<Щ	�oL��AJ߸
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
input/imagesPlaceholder"/device:GPU:0*$
shape:���������@@*
dtype0*/
_output_shapes
:���������@@
�
input/correct_labelsPlaceholder"/device:GPU:0*
shape:���������*
dtype0*'
_output_shapes
:���������
g
input/training_flagPlaceholder"/device:GPU:0*
dtype0
*
_output_shapes
:*
shape:
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
-CNN1/weights/truncated_normal/TruncatedNormalTruncatedNormal#CNN1/weights/truncated_normal/shape"/device:GPU:0*

seed *
T0*
dtype0*&
_output_shapes
:*
seed2 
�
!CNN1/weights/truncated_normal/mulMul-CNN1/weights/truncated_normal/TruncatedNormal$CNN1/weights/truncated_normal/stddev"/device:GPU:0*
T0*&
_output_shapes
:
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
CNN1/weights/Variable/AssignAssignCNN1/weights/VariableCNN1/weights/truncated_normal"/device:GPU:0*(
_class
loc:@CNN1/weights/Variable*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
�
CNN1/weights/Variable/readIdentityCNN1/weights/Variable"/device:GPU:0*(
_class
loc:@CNN1/weights/Variable*&
_output_shapes
:*
T0
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
"CNN1/weights/summaries/range/deltaConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
CNN1/weights/summaries/rangeRange"CNN1/weights/summaries/range/startCNN1/weights/summaries/Rank"CNN1/weights/summaries/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/weights/summaries/MeanMeanCNN1/weights/Variable/readCNN1/weights/summaries/range"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
 CNN1/weights/summaries/mean/tagsConst"/device:GPU:0*
_output_shapes
: *,
value#B! BCNN1/weights/summaries/mean*
dtype0
�
CNN1/weights/summaries/meanScalarSummary CNN1/weights/summaries/mean/tagsCNN1/weights/summaries/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
!CNN1/weights/summaries/stddev/subSubCNN1/weights/Variable/readCNN1/weights/summaries/Mean"/device:GPU:0*
T0*&
_output_shapes
:
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
"CNN1/weights/summaries/stddev/MeanMean$CNN1/weights/summaries/stddev/Square#CNN1/weights/summaries/stddev/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
CNN1/weights/summaries/stddev_1ScalarSummary$CNN1/weights/summaries/stddev_1/tags"CNN1/weights/summaries/stddev/Sqrt"/device:GPU:0*
T0*
_output_shapes
: 
n
CNN1/weights/summaries/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
u
$CNN1/weights/summaries/range_1/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
u
$CNN1/weights/summaries/range_1/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN1/weights/summaries/range_1Range$CNN1/weights/summaries/range_1/startCNN1/weights/summaries/Rank_1$CNN1/weights/summaries/range_1/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
�
CNN1/weights/summaries/MaxMaxCNN1/weights/Variable/readCNN1/weights/summaries/range_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
CNN1/weights/summaries/MinMinCNN1/weights/Variable/readCNN1/weights/summaries/range_2"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
CNN1/weights/summaries/min/tagsConst"/device:GPU:0*+
value"B  BCNN1/weights/summaries/min*
dtype0*
_output_shapes
: 
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
CNN1/biases/summaries/rangeRange!CNN1/biases/summaries/range/startCNN1/biases/summaries/Rank!CNN1/biases/summaries/range/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN1/biases/summaries/MeanMeanCNN1/biases/Variable/readCNN1/biases/summaries/range"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN1/biases/summaries/mean/tagsConst"/device:GPU:0*+
value"B  BCNN1/biases/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/meanScalarSummaryCNN1/biases/summaries/mean/tagsCNN1/biases/summaries/Mean"/device:GPU:0*
T0*
_output_shapes
: 
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
!CNN1/biases/summaries/stddev/MeanMean#CNN1/biases/summaries/stddev/Square"CNN1/biases/summaries/stddev/Const"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
CNN1/biases/summaries/MaxMaxCNN1/biases/Variable/readCNN1/biases/summaries/range_1"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
#CNN1/biases/summaries/range_2/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
t
#CNN1/biases/summaries/range_2/deltaConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/range_2Range#CNN1/biases/summaries/range_2/startCNN1/biases/summaries/Rank_2#CNN1/biases/summaries/range_2/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
�
CNN1/biases/summaries/MinMinCNN1/biases/Variable/readCNN1/biases/summaries/range_2"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN1/biases/summaries/min/tagsConst"/device:GPU:0**
value!B BCNN1/biases/summaries/min*
dtype0*
_output_shapes
: 
�
CNN1/biases/summaries/minScalarSummaryCNN1/biases/summaries/min/tagsCNN1/biases/summaries/Min"/device:GPU:0*
T0*
_output_shapes
: 
�
#CNN1/biases/summaries/histogram/tagConst"/device:GPU:0*
_output_shapes
: *0
value'B% BCNN1/biases/summaries/histogram*
dtype0
�
CNN1/biases/summaries/histogramHistogramSummary#CNN1/biases/summaries/histogram/tagCNN1/biases/Variable/read"/device:GPU:0*
T0*
_output_shapes
: 
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
CNN1/addAddCNN1/Conv2DCNN1/biases/Variable/read"/device:GPU:0*
T0*/
_output_shapes
:���������@@
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
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
:@*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
	container *
shape:@
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
batch_normalization/gamma/readIdentitybatch_normalization/gamma"/device:GPU:0*
T0*,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
:@
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
shape:@*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta*
	container 
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
VariableV2"/device:GPU:0*
shared_name *2
_class(
&$loc:@batch_normalization/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
&batch_normalization/moving_mean/AssignAssignbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros"/device:GPU:0*
_output_shapes
:@*
use_locking(*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
validate_shape(
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
VariableV2"/device:GPU:0*6
_class,
*(loc:@batch_normalization/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
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
(batch_normalization/moving_variance/readIdentity#batch_normalization/moving_variance"/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
$CNN1/batch_normalization/cond/SwitchSwitchinput/training_flaginput/training_flag"/device:GPU:0*
_output_shapes

::*
T0

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
x
%CNN1/batch_normalization/cond/pred_idIdentityinput/training_flag"/device:GPU:0*
T0
*
_output_shapes
:
�
#CNN1/batch_normalization/cond/ConstConst'^CNN1/batch_normalization/cond/switch_t"/device:GPU:0*
_output_shapes
: *
valueB *
dtype0
�
%CNN1/batch_normalization/cond/Const_1Const'^CNN1/batch_normalization/cond/switch_t"/device:GPU:0*
_output_shapes
: *
valueB *
dtype0
�
3CNN1/batch_normalization/cond/FusedBatchNorm/SwitchSwitch	CNN1/Relu%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*
T0*
_class
loc:@CNN1/Relu*J
_output_shapes8
6:���������@@:���������@@
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
,CNN1/batch_normalization/cond/FusedBatchNormFusedBatchNorm5CNN1/batch_normalization/cond/FusedBatchNorm/Switch:17CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1:17CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2:1#CNN1/batch_normalization/cond/Const%CNN1/batch_normalization/cond/Const_1"/device:GPU:0*G
_output_shapes5
3:���������@@:@:@:@:@*
is_training(*
epsilon%o�:*
T0*
data_formatNCHW
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
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch$batch_normalization/moving_mean/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
:@:@*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch(batch_normalization/moving_variance/read%CNN1/batch_normalization/cond/pred_id"/device:GPU:0*6
_class,
*(loc:@batch_normalization/moving_variance* 
_output_shapes
:@:@*
T0
�
.CNN1/batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm5CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_27CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*G
_output_shapes5
3:���������@@:@:@:@:@*
is_training( *
epsilon%o�:*
T0*
data_formatNCHW
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
+CNN1/batch_normalization/ExpandDims_1/inputConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
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
 CNN1/batch_normalization/ReshapeReshapeinput/training_flag&CNN1/batch_normalization/Reshape/shape"/device:GPU:0*
T0
*
Tshape0*
_output_shapes
:
�
CNN1/batch_normalization/SelectSelect CNN1/batch_normalization/Reshape#CNN1/batch_normalization/ExpandDims%CNN1/batch_normalization/ExpandDims_1"/device:GPU:0*
T0*
_output_shapes
:
�
 CNN1/batch_normalization/SqueezeSqueezeCNN1/batch_normalization/Select"/device:GPU:0*
squeeze_dims
 *
T0*
_output_shapes
: 
�
-CNN1/batch_normalization/AssignMovingAvg/readIdentitybatch_normalization/moving_mean"/device:GPU:0*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
�
,CNN1/batch_normalization/AssignMovingAvg/SubSub-CNN1/batch_normalization/AssignMovingAvg/read%CNN1/batch_normalization/cond/Merge_1"/device:GPU:0*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:@
�
,CNN1/batch_normalization/AssignMovingAvg/MulMul,CNN1/batch_normalization/AssignMovingAvg/Sub CNN1/batch_normalization/Squeeze"/device:GPU:0*
_output_shapes
:@*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
(CNN1/batch_normalization/AssignMovingAvg	AssignSubbatch_normalization/moving_mean,CNN1/batch_normalization/AssignMovingAvg/Mul"/device:GPU:0*
_output_shapes
:@*
use_locking( *
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
/CNN1/batch_normalization/AssignMovingAvg_1/readIdentity#batch_normalization/moving_variance"/device:GPU:0*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@
�
.CNN1/batch_normalization/AssignMovingAvg_1/SubSub/CNN1/batch_normalization/AssignMovingAvg_1/read%CNN1/batch_normalization/cond/Merge_2"/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
.CNN1/batch_normalization/AssignMovingAvg_1/MulMul.CNN1/batch_normalization/AssignMovingAvg_1/Sub CNN1/batch_normalization/Squeeze"/device:GPU:0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:@*
T0
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
$CNN2/weights/truncated_normal/stddevConst"/device:GPU:0*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
-CNN2/weights/truncated_normal/TruncatedNormalTruncatedNormal#CNN2/weights/truncated_normal/shape"/device:GPU:0*

seed *
T0*
dtype0*&
_output_shapes
: *
seed2 
�
!CNN2/weights/truncated_normal/mulMul-CNN2/weights/truncated_normal/TruncatedNormal$CNN2/weights/truncated_normal/stddev"/device:GPU:0*&
_output_shapes
: *
T0
�
CNN2/weights/truncated_normalAdd!CNN2/weights/truncated_normal/mul"CNN2/weights/truncated_normal/mean"/device:GPU:0*
T0*&
_output_shapes
: 
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
CNN2/weights/Variable/AssignAssignCNN2/weights/VariableCNN2/weights/truncated_normal"/device:GPU:0*
use_locking(*
T0*(
_class
loc:@CNN2/weights/Variable*
validate_shape(*&
_output_shapes
: 
�
CNN2/weights/Variable/readIdentityCNN2/weights/Variable"/device:GPU:0*&
_output_shapes
: *
T0*(
_class
loc:@CNN2/weights/Variable
l
CNN2/weights/summaries/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
s
"CNN2/weights/summaries/range/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
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
CNN2/weights/summaries/MeanMeanCNN2/weights/Variable/readCNN2/weights/summaries/range"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
 CNN2/weights/summaries/mean/tagsConst"/device:GPU:0*,
value#B! BCNN2/weights/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN2/weights/summaries/meanScalarSummary CNN2/weights/summaries/mean/tagsCNN2/weights/summaries/Mean"/device:GPU:0*
T0*
_output_shapes
: 
�
!CNN2/weights/summaries/stddev/subSubCNN2/weights/Variable/readCNN2/weights/summaries/Mean"/device:GPU:0*
T0*&
_output_shapes
: 
�
$CNN2/weights/summaries/stddev/SquareSquare!CNN2/weights/summaries/stddev/sub"/device:GPU:0*&
_output_shapes
: *
T0
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
CNN2/weights/summaries/range_2Range$CNN2/weights/summaries/range_2/startCNN2/weights/summaries/Rank_2$CNN2/weights/summaries/range_2/delta"/device:GPU:0*
_output_shapes
:*

Tidx0
�
CNN2/weights/summaries/MinMinCNN2/weights/Variable/readCNN2/weights/summaries/range_2"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
$CNN2/weights/summaries/histogram/tagConst"/device:GPU:0*1
value(B& B CNN2/weights/summaries/histogram*
dtype0*
_output_shapes
: 
�
 CNN2/weights/summaries/histogramHistogramSummary$CNN2/weights/summaries/histogram/tagCNN2/weights/Variable/read"/device:GPU:0*
_output_shapes
: *
T0
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
#CNN2/biases/summaries/stddev/SquareSquare CNN2/biases/summaries/stddev/sub"/device:GPU:0*
_output_shapes
: *
T0
{
"CNN2/biases/summaries/stddev/ConstConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
!CNN2/biases/summaries/stddev/MeanMean#CNN2/biases/summaries/stddev/Square"CNN2/biases/summaries/stddev/Const"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
CNN2/biases/summaries/stddev_1ScalarSummary#CNN2/biases/summaries/stddev_1/tags!CNN2/biases/summaries/stddev/Sqrt"/device:GPU:0*
T0*
_output_shapes
: 
m
CNN2/biases/summaries/Rank_1Const"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
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
CNN2/biases/summaries/MaxMaxCNN2/biases/Variable/readCNN2/biases/summaries/range_1"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
CNN2/biases/summaries/MinMinCNN2/biases/Variable/readCNN2/biases/summaries/range_2"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
CNN2/biases/summaries/histogramHistogramSummary#CNN2/biases/summaries/histogram/tagCNN2/biases/Variable/read"/device:GPU:0*
T0*
_output_shapes
: 
�
CNN2/Conv2DConv2D#CNN1/batch_normalization/cond/MergeCNN2/weights/Variable/read"/device:GPU:0*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:���������   
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
"batch_normalization_1/gamma/AssignAssignbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones"/device:GPU:0*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
validate_shape(
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
shape: *
dtype0*
_output_shapes
: *
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container 
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
batch_normalization_1/beta/readIdentitybatch_normalization_1/beta"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: 
�
3batch_normalization_1/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_1/moving_mean*
valueB *    
�
!batch_normalization_1/moving_mean
VariableV2"/device:GPU:0*
shared_name *4
_class*
(&loc:@batch_normalization_1/moving_mean*
	container *
shape: *
dtype0*
_output_shapes
: 
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
&batch_normalization_1/moving_mean/readIdentity!batch_normalization_1/moving_mean"/device:GPU:0*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
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
VariableV2"/device:GPU:0*
shared_name *8
_class.
,*loc:@batch_normalization_1/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: 
�
,batch_normalization_1/moving_variance/AssignAssign%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones"/device:GPU:0*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes
: 
�
*batch_normalization_1/moving_variance/readIdentity%batch_normalization_1/moving_variance"/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
�
$CNN2/batch_normalization/cond/SwitchSwitchinput/training_flaginput/training_flag"/device:GPU:0*
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
x
%CNN2/batch_normalization/cond/pred_idIdentityinput/training_flag"/device:GPU:0*
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
5CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1Switch batch_normalization_1/gamma/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
: : 
�
5CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_1/beta* 
_output_shapes
: : 
�
,CNN2/batch_normalization/cond/FusedBatchNormFusedBatchNorm5CNN2/batch_normalization/cond/FusedBatchNorm/Switch:17CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1:17CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2:1#CNN2/batch_normalization/cond/Const%CNN2/batch_normalization/cond/Const_1"/device:GPU:0*
data_formatNCHW*G
_output_shapes5
3:���������   : : : : *
is_training(*
epsilon%o�:*
T0
�
5CNN2/batch_normalization/cond/FusedBatchNorm_1/SwitchSwitch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*J
_output_shapes8
6:���������   :���������   *
T0*
_class
loc:@CNN2/Relu
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_1/gamma/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*.
_class$
" loc:@batch_normalization_1/gamma* 
_output_shapes
: : *
T0
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
: : *
T0*-
_class#
!loc:@batch_normalization_1/beta
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_1/moving_mean/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*4
_class*
(&loc:@batch_normalization_1/moving_mean* 
_output_shapes
: : *
T0
�
7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_1/moving_variance/read%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance* 
_output_shapes
: : 
�
.CNN2/batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm5CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_27CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
epsilon%o�:*
T0*
data_formatNCHW*G
_output_shapes5
3:���������   : : : : *
is_training( 
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
'CNN2/batch_normalization/ExpandDims/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
#CNN2/batch_normalization/ExpandDims
ExpandDims)CNN2/batch_normalization/ExpandDims/input'CNN2/batch_normalization/ExpandDims/dim"/device:GPU:0*
T0*
_output_shapes
:*

Tdim0

+CNN2/batch_normalization/ExpandDims_1/inputConst"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
z
)CNN2/batch_normalization/ExpandDims_1/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
%CNN2/batch_normalization/ExpandDims_1
ExpandDims+CNN2/batch_normalization/ExpandDims_1/input)CNN2/batch_normalization/ExpandDims_1/dim"/device:GPU:0*
_output_shapes
:*

Tdim0*
T0

&CNN2/batch_normalization/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
 CNN2/batch_normalization/ReshapeReshapeinput/training_flag&CNN2/batch_normalization/Reshape/shape"/device:GPU:0*
Tshape0*
_output_shapes
:*
T0

�
CNN2/batch_normalization/SelectSelect CNN2/batch_normalization/Reshape#CNN2/batch_normalization/ExpandDims%CNN2/batch_normalization/ExpandDims_1"/device:GPU:0*
_output_shapes
:*
T0
�
 CNN2/batch_normalization/SqueezeSqueezeCNN2/batch_normalization/Select"/device:GPU:0*
T0*
_output_shapes
: *
squeeze_dims
 
�
-CNN2/batch_normalization/AssignMovingAvg/readIdentity!batch_normalization_1/moving_mean"/device:GPU:0*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
�
,CNN2/batch_normalization/AssignMovingAvg/SubSub-CNN2/batch_normalization/AssignMovingAvg/read%CNN2/batch_normalization/cond/Merge_1"/device:GPU:0*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: 
�
,CNN2/batch_normalization/AssignMovingAvg/MulMul,CNN2/batch_normalization/AssignMovingAvg/Sub CNN2/batch_normalization/Squeeze"/device:GPU:0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: *
T0
�
(CNN2/batch_normalization/AssignMovingAvg	AssignSub!batch_normalization_1/moving_mean,CNN2/batch_normalization/AssignMovingAvg/Mul"/device:GPU:0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
/CNN2/batch_normalization/AssignMovingAvg_1/readIdentity%batch_normalization_1/moving_variance"/device:GPU:0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: *
T0
�
.CNN2/batch_normalization/AssignMovingAvg_1/SubSub/CNN2/batch_normalization/AssignMovingAvg_1/read%CNN2/batch_normalization/cond/Merge_2"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
�
.CNN2/batch_normalization/AssignMovingAvg_1/MulMul.CNN2/batch_normalization/AssignMovingAvg_1/Sub CNN2/batch_normalization/Squeeze"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
�
*CNN2/batch_normalization/AssignMovingAvg_1	AssignSub%batch_normalization_1/moving_variance.CNN2/batch_normalization/AssignMovingAvg_1/Mul"/device:GPU:0*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: 
r
CNN2/batch_norm/tagConst"/device:GPU:0*
_output_shapes
: * 
valueB BCNN2/batch_norm*
dtype0
�
CNN2/batch_normHistogramSummaryCNN2/batch_norm/tag#CNN2/batch_normalization/cond/Merge"/device:GPU:0*
_output_shapes
: *
T0
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
-CNN3/weights/truncated_normal/TruncatedNormalTruncatedNormal#CNN3/weights/truncated_normal/shape"/device:GPU:0*

seed *
T0*
dtype0*&
_output_shapes
: @*
seed2 
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
CNN3/weights/Variable/AssignAssignCNN3/weights/VariableCNN3/weights/truncated_normal"/device:GPU:0*&
_output_shapes
: @*
use_locking(*
T0*(
_class
loc:@CNN3/weights/Variable*
validate_shape(
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
"CNN3/weights/summaries/range/startConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
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
CNN3/weights/summaries/MeanMeanCNN3/weights/Variable/readCNN3/weights/summaries/range"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
!CNN3/weights/summaries/stddev/subSubCNN3/weights/Variable/readCNN3/weights/summaries/Mean"/device:GPU:0*&
_output_shapes
: @*
T0
�
$CNN3/weights/summaries/stddev/SquareSquare!CNN3/weights/summaries/stddev/sub"/device:GPU:0*&
_output_shapes
: @*
T0
�
#CNN3/weights/summaries/stddev/ConstConst"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
"CNN3/weights/summaries/stddev/MeanMean$CNN3/weights/summaries/stddev/Square#CNN3/weights/summaries/stddev/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
~
"CNN3/weights/summaries/stddev/SqrtSqrt"CNN3/weights/summaries/stddev/Mean"/device:GPU:0*
_output_shapes
: *
T0
�
$CNN3/weights/summaries/stddev_1/tagsConst"/device:GPU:0*0
value'B% BCNN3/weights/summaries/stddev_1*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/stddev_1ScalarSummary$CNN3/weights/summaries/stddev_1/tags"CNN3/weights/summaries/stddev/Sqrt"/device:GPU:0*
T0*
_output_shapes
: 
n
CNN3/weights/summaries/Rank_1Const"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
u
$CNN3/weights/summaries/range_1/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
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
CNN3/weights/summaries/maxScalarSummaryCNN3/weights/summaries/max/tagsCNN3/weights/summaries/Max"/device:GPU:0*
_output_shapes
: *
T0
n
CNN3/weights/summaries/Rank_2Const"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
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
CNN3/weights/summaries/MinMinCNN3/weights/Variable/readCNN3/weights/summaries/range_2"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
CNN3/weights/summaries/min/tagsConst"/device:GPU:0*+
value"B  BCNN3/weights/summaries/min*
dtype0*
_output_shapes
: 
�
CNN3/weights/summaries/minScalarSummaryCNN3/weights/summaries/min/tagsCNN3/weights/summaries/Min"/device:GPU:0*
_output_shapes
: *
T0
�
$CNN3/weights/summaries/histogram/tagConst"/device:GPU:0*
_output_shapes
: *1
value(B& B CNN3/weights/summaries/histogram*
dtype0
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
VariableV2"/device:GPU:0*
_output_shapes
:@*
	container *
shape:@*
shared_name *
dtype0
�
CNN3/biases/Variable/AssignAssignCNN3/biases/VariableCNN3/biases/Const"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN3/biases/Variable*
validate_shape(*
_output_shapes
:@
�
CNN3/biases/Variable/readIdentityCNN3/biases/Variable"/device:GPU:0*
T0*'
_class
loc:@CNN3/biases/Variable*
_output_shapes
:@
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
CNN3/biases/summaries/MeanMeanCNN3/biases/Variable/readCNN3/biases/summaries/range"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
CNN3/biases/summaries/mean/tagsConst"/device:GPU:0*+
value"B  BCNN3/biases/summaries/mean*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/meanScalarSummaryCNN3/biases/summaries/mean/tagsCNN3/biases/summaries/Mean"/device:GPU:0*
_output_shapes
: *
T0
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
!CNN3/biases/summaries/stddev/MeanMean#CNN3/biases/summaries/stddev/Square"CNN3/biases/summaries/stddev/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
CNN3/biases/summaries/stddev_1ScalarSummary#CNN3/biases/summaries/stddev_1/tags!CNN3/biases/summaries/stddev/Sqrt"/device:GPU:0*
_output_shapes
: *
T0
m
CNN3/biases/summaries/Rank_1Const"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
t
#CNN3/biases/summaries/range_1/startConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
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
CNN3/biases/summaries/MaxMaxCNN3/biases/Variable/readCNN3/biases/summaries/range_1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
CNN3/biases/summaries/max/tagsConst"/device:GPU:0**
value!B BCNN3/biases/summaries/max*
dtype0*
_output_shapes
: 
�
CNN3/biases/summaries/maxScalarSummaryCNN3/biases/summaries/max/tagsCNN3/biases/summaries/Max"/device:GPU:0*
T0*
_output_shapes
: 
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
CNN3/biases/summaries/range_2Range#CNN3/biases/summaries/range_2/startCNN3/biases/summaries/Rank_2#CNN3/biases/summaries/range_2/delta"/device:GPU:0*

Tidx0*
_output_shapes
:
�
CNN3/biases/summaries/MinMinCNN3/biases/Variable/readCNN3/biases/summaries/range_2"/device:GPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
CNN3/biases/summaries/histogramHistogramSummary#CNN3/biases/summaries/histogram/tagCNN3/biases/Variable/read"/device:GPU:0*
_output_shapes
: *
T0
�
CNN3/Conv2DConv2D#CNN2/batch_normalization/cond/MergeCNN3/weights/Variable/read"/device:GPU:0*/
_output_shapes
:���������@*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
CNN3/addAddCNN3/Conv2DCNN3/biases/Variable/read"/device:GPU:0*/
_output_shapes
:���������@*
T0
d
	CNN3/ReluReluCNN3/add"/device:GPU:0*
T0*/
_output_shapes
:���������@
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
"batch_normalization_2/gamma/AssignAssignbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(
�
 batch_normalization_2/gamma/readIdentitybatch_normalization_2/gamma"/device:GPU:0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:*
T0
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
batch_normalization_2/beta/readIdentitybatch_normalization_2/beta"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
:
�
3batch_normalization_2/moving_mean/Initializer/zerosConst*
_output_shapes
:*4
_class*
(&loc:@batch_normalization_2/moving_mean*
valueB*    *
dtype0
�
!batch_normalization_2/moving_mean
VariableV2"/device:GPU:0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
&batch_normalization_2/moving_mean/readIdentity!batch_normalization_2/moving_mean"/device:GPU:0*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
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
$CNN3/batch_normalization/cond/SwitchSwitchinput/training_flaginput/training_flag"/device:GPU:0*
T0
*
_output_shapes

::
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
x
%CNN3/batch_normalization/cond/pred_idIdentityinput/training_flag"/device:GPU:0*
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
,CNN3/batch_normalization/cond/FusedBatchNormFusedBatchNorm5CNN3/batch_normalization/cond/FusedBatchNorm/Switch:17CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1:17CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2:1#CNN3/batch_normalization/cond/Const%CNN3/batch_normalization/cond/Const_1"/device:GPU:0*
data_formatNCHW*G
_output_shapes5
3:���������@::::*
is_training(*
epsilon%o�:*
T0
�
5CNN3/batch_normalization/cond/FusedBatchNorm_1/SwitchSwitch	CNN3/Relu%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*
T0*
_class
loc:@CNN3/Relu*J
_output_shapes8
6:���������@:���������@
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
::*
T0*.
_class$
" loc:@batch_normalization_2/gamma
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2Switchbatch_normalization_2/beta/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*-
_class#
!loc:@batch_normalization_2/beta* 
_output_shapes
::*
T0
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_3Switch&batch_normalization_2/moving_mean/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0* 
_output_shapes
::*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4Switch*batch_normalization_2/moving_variance/read%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*8
_class.
,*loc:@batch_normalization_2/moving_variance* 
_output_shapes
::*
T0
�
.CNN3/batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm5CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch7CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_27CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
epsilon%o�:*
T0*
data_formatNCHW*G
_output_shapes5
3:���������@::::*
is_training( 
�
#CNN3/batch_normalization/cond/MergeMerge.CNN3/batch_normalization/cond/FusedBatchNorm_1,CNN3/batch_normalization/cond/FusedBatchNorm"/device:GPU:0*
T0*
N*1
_output_shapes
:���������@: 
�
%CNN3/batch_normalization/cond/Merge_1Merge0CNN3/batch_normalization/cond/FusedBatchNorm_1:1.CNN3/batch_normalization/cond/FusedBatchNorm:1"/device:GPU:0*
_output_shapes

:: *
T0*
N
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
'CNN3/batch_normalization/ExpandDims/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
#CNN3/batch_normalization/ExpandDims
ExpandDims)CNN3/batch_normalization/ExpandDims/input'CNN3/batch_normalization/ExpandDims/dim"/device:GPU:0*
_output_shapes
:*

Tdim0*
T0

+CNN3/batch_normalization/ExpandDims_1/inputConst"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
z
)CNN3/batch_normalization/ExpandDims_1/dimConst"/device:GPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
%CNN3/batch_normalization/ExpandDims_1
ExpandDims+CNN3/batch_normalization/ExpandDims_1/input)CNN3/batch_normalization/ExpandDims_1/dim"/device:GPU:0*
_output_shapes
:*

Tdim0*
T0

&CNN3/batch_normalization/Reshape/shapeConst"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
 CNN3/batch_normalization/ReshapeReshapeinput/training_flag&CNN3/batch_normalization/Reshape/shape"/device:GPU:0*
Tshape0*
_output_shapes
:*
T0

�
CNN3/batch_normalization/SelectSelect CNN3/batch_normalization/Reshape#CNN3/batch_normalization/ExpandDims%CNN3/batch_normalization/ExpandDims_1"/device:GPU:0*
_output_shapes
:*
T0
�
 CNN3/batch_normalization/SqueezeSqueezeCNN3/batch_normalization/Select"/device:GPU:0*
T0*
_output_shapes
: *
squeeze_dims
 
�
-CNN3/batch_normalization/AssignMovingAvg/readIdentity!batch_normalization_2/moving_mean"/device:GPU:0*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
�
,CNN3/batch_normalization/AssignMovingAvg/SubSub-CNN3/batch_normalization/AssignMovingAvg/read%CNN3/batch_normalization/cond/Merge_1"/device:GPU:0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
:*
T0
�
,CNN3/batch_normalization/AssignMovingAvg/MulMul,CNN3/batch_normalization/AssignMovingAvg/Sub CNN3/batch_normalization/Squeeze"/device:GPU:0*
_output_shapes
:*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
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
.CNN3/batch_normalization/AssignMovingAvg_1/SubSub/CNN3/batch_normalization/AssignMovingAvg_1/read%CNN3/batch_normalization/cond/Merge_2"/device:GPU:0*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
�
.CNN3/batch_normalization/AssignMovingAvg_1/MulMul.CNN3/batch_normalization/AssignMovingAvg_1/Sub CNN3/batch_normalization/Squeeze"/device:GPU:0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:*
T0
�
*CNN3/batch_normalization/AssignMovingAvg_1	AssignSub%batch_normalization_2/moving_variance.CNN3/batch_normalization/AssignMovingAvg_1/Mul"/device:GPU:0*
_output_shapes
:*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance
r
CNN3/batch_norm/tagConst"/device:GPU:0* 
valueB BCNN3/batch_norm*
dtype0*
_output_shapes
: 
�
CNN3/batch_normHistogramSummaryCNN3/batch_norm/tag#CNN3/batch_normalization/cond/Merge"/device:GPU:0*
_output_shapes
: *
T0
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
FC1/truncated_normal/stddevConst"/device:GPU:0*
valueB
 *���=*
dtype0*
_output_shapes
: 
�
$FC1/truncated_normal/TruncatedNormalTruncatedNormalFC1/truncated_normal/shape"/device:GPU:0*

seed *
T0*
dtype0*!
_output_shapes
:���*
seed2 
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
FC1/Variable_1/readIdentityFC1/Variable_1"/device:GPU:0*
_output_shapes	
:�*
T0*!
_class
loc:@FC1/Variable_1
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

FC1/MatMulMatMulFC1/ReshapeFC1/Variable/read"/device:GPU:0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
q
FC1/addAdd
FC1/MatMulFC1/Variable_1/read"/device:GPU:0*
T0*(
_output_shapes
:����������
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
VariableV2"/device:GPU:0*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *.
_class$
" loc:@batch_normalization_3/gamma*
	container 
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
 batch_normalization_3/gamma/readIdentitybatch_normalization_3/gamma"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:�
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
3batch_normalization_3/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
!batch_normalization_3/moving_mean
VariableV2"/device:GPU:0*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@batch_normalization_3/moving_mean*
	container *
shape:�
�
(batch_normalization_3/moving_mean/AssignAssign!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros"/device:GPU:0*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
validate_shape(
�
&batch_normalization_3/moving_mean/readIdentity!batch_normalization_3/moving_mean"/device:GPU:0*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
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
6FC1/batch_normalization/moments/mean/reduction_indicesConst"/device:GPU:0*
_output_shapes
:*
valueB: *
dtype0
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
'FC1/batch_normalization/moments/SqueezeSqueeze$FC1/batch_normalization/moments/mean"/device:GPU:0*
_output_shapes	
:�*
squeeze_dims
 *
T0
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
FC1/batch_normalization/ReshapeReshapeinput/training_flag%FC1/batch_normalization/Reshape/shape"/device:GPU:0*
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
!FC1/batch_normalization/Reshape_1Reshapeinput/training_flag'FC1/batch_normalization/Reshape_1/shape"/device:GPU:0*
_output_shapes
:*
T0
*
Tshape0
�
 FC1/batch_normalization/Select_1Select!FC1/batch_normalization/Reshape_1$FC1/batch_normalization/ExpandDims_2$FC1/batch_normalization/ExpandDims_3"/device:GPU:0*
T0*
_output_shapes
:	�
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
(FC1/batch_normalization/ExpandDims_4/dimConst"/device:GPU:0*
_output_shapes
: *
value	B : *
dtype0
�
$FC1/batch_normalization/ExpandDims_4
ExpandDims*FC1/batch_normalization/ExpandDims_4/input(FC1/batch_normalization/ExpandDims_4/dim"/device:GPU:0*

Tdim0*
T0*
_output_shapes
:
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
!FC1/batch_normalization/Reshape_2Reshapeinput/training_flag'FC1/batch_normalization/Reshape_2/shape"/device:GPU:0*
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
!FC1/batch_normalization/Squeeze_2Squeeze FC1/batch_normalization/Select_2"/device:GPU:0*
T0*
_output_shapes
: *
squeeze_dims
 
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
+FC1/batch_normalization/AssignMovingAvg/subSub-FC1/batch_normalization/AssignMovingAvg/sub/x!FC1/batch_normalization/Squeeze_2"/device:GPU:0*
_output_shapes
: *
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
�
-FC1/batch_normalization/AssignMovingAvg/sub_1Sub&batch_normalization_3/moving_mean/readFC1/batch_normalization/Squeeze"/device:GPU:0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes	
:�*
T0
�
+FC1/batch_normalization/AssignMovingAvg/mulMul-FC1/batch_normalization/AssignMovingAvg/sub_1+FC1/batch_normalization/AssignMovingAvg/sub"/device:GPU:0*
_output_shapes	
:�*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean
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
-FC1/batch_normalization/AssignMovingAvg_1/subSub/FC1/batch_normalization/AssignMovingAvg_1/sub/x!FC1/batch_normalization/Squeeze_2"/device:GPU:0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
: *
T0
�
/FC1/batch_normalization/AssignMovingAvg_1/sub_1Sub*batch_normalization_3/moving_variance/read!FC1/batch_normalization/Squeeze_1"/device:GPU:0*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
-FC1/batch_normalization/AssignMovingAvg_1/mulMul/FC1/batch_normalization/AssignMovingAvg_1/sub_1-FC1/batch_normalization/AssignMovingAvg_1/sub"/device:GPU:0*
_output_shapes	
:�*
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
�
)FC1/batch_normalization/AssignMovingAvg_1	AssignSub%batch_normalization_3/moving_variance-FC1/batch_normalization/AssignMovingAvg_1/mul"/device:GPU:0*
_output_shapes	
:�*
use_locking( *
T0*8
_class.
,*loc:@batch_normalization_3/moving_variance
{
'FC1/batch_normalization/batchnorm/add/yConst"/device:GPU:0*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
%FC1/batch_normalization/batchnorm/addAdd!FC1/batch_normalization/Squeeze_1'FC1/batch_normalization/batchnorm/add/y"/device:GPU:0*
_output_shapes	
:�*
T0
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
e
dropout/keep_probPlaceholder"/device:GPU:0*
dtype0*
_output_shapes
:*
shape:
~
Readout/truncated_normal/shapeConst"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0
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
Readout/truncated_normalAddReadout/truncated_normal/mulReadout/truncated_normal/mean"/device:GPU:0*
_output_shapes
:	�*
T0
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
Readout/ConstConst"/device:GPU:0*
_output_shapes
:*
valueB*���=*
dtype0
�
Readout/Variable_1
VariableV2"/device:GPU:0*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
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
Readout/Variable_1/readIdentityReadout/Variable_1"/device:GPU:0*%
_class
loc:@Readout/Variable_1*
_output_shapes
:*
T0
�
Readout/MatMulMatMul'FC1/batch_normalization/batchnorm/add_1Readout/Variable/read"/device:GPU:0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
Readout/predicted_labelsAddReadout/MatMulReadout/Variable_1/read"/device:GPU:0*'
_output_shapes
:���������*
T0
t
#Readout/predicted_results/dimensionConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
Readout/predicted_resultsArgMaxReadout/predicted_labels#Readout/predicted_results/dimension"/device:GPU:0*#
_output_shapes
:���������*

Tidx0*
T0*
output_type0	
i
cross_entropy_total/RankConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 
�
cross_entropy_total/ShapeShapeReadout/predicted_labels"/device:GPU:0*
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
�
cross_entropy_total/Shape_1ShapeReadout/predicted_labels"/device:GPU:0*
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
cross_entropy_total/ReshapeReshapeReadout/predicted_labelscross_entropy_total/concat"/device:GPU:0*
T0*
Tshape0*0
_output_shapes
:������������������
k
cross_entropy_total/Rank_2Const"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
~
cross_entropy_total/Shape_2Shapeinput/correct_labels"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
l
cross_entropy_total/Sub_1/yConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
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
cross_entropy_total/concat_1ConcatV2%cross_entropy_total/concat_1/values_0cross_entropy_total/Slice_1!cross_entropy_total/concat_1/axis"/device:GPU:0*
N*
_output_shapes
:*

Tidx0*
T0
�
cross_entropy_total/Reshape_1Reshapeinput/correct_labelscross_entropy_total/concat_1"/device:GPU:0*
Tshape0*0
_output_shapes
:������������������*
T0
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
cross_entropy_total/Sub_2Subcross_entropy_total/Rankcross_entropy_total/Sub_2/y"/device:GPU:0*
T0*
_output_shapes
: 
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
cross_entropy_total/Reshape_2Reshape1cross_entropy_total/SoftmaxCrossEntropyWithLogitscross_entropy_total/Slice_2"/device:GPU:0*
Tshape0*#
_output_shapes
:���������*
T0
r
cross_entropy_total/ConstConst"/device:GPU:0*
valueB: *
dtype0*
_output_shapes
:
�
cross_entropy_total/MeanMeancross_entropy_total/Reshape_2cross_entropy_total/Const"/device:GPU:0*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
;train/gradients/cross_entropy_total/Mean_grad/Reshape/shapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/cross_entropy_total/Mean_grad/ReshapeReshapetrain/gradients/Fill;train/gradients/cross_entropy_total/Mean_grad/Reshape/shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:
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
2train/gradients/cross_entropy_total/Mean_grad/ProdProd5train/gradients/cross_entropy_total/Mean_grad/Shape_13train/gradients/cross_entropy_total/Mean_grad/Const"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
_output_shapes
: 
�
5train/gradients/cross_entropy_total/Mean_grad/Const_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
dtype0*
_output_shapes
:
�
4train/gradients/cross_entropy_total/Mean_grad/Prod_1Prod5train/gradients/cross_entropy_total/Mean_grad/Shape_25train/gradients/cross_entropy_total/Mean_grad/Const_1"/device:GPU:0*
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
7train/gradients/cross_entropy_total/Mean_grad/Maximum/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1*
dtype0*
_output_shapes
: 
�
5train/gradients/cross_entropy_total/Mean_grad/MaximumMaximum4train/gradients/cross_entropy_total/Mean_grad/Prod_17train/gradients/cross_entropy_total/Mean_grad/Maximum/y"/device:GPU:0*
_output_shapes
: *
T0*H
_class>
<:loc:@train/gradients/cross_entropy_total/Mean_grad/Shape_1
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
5train/gradients/cross_entropy_total/Mean_grad/truedivRealDiv2train/gradients/cross_entropy_total/Mean_grad/Tile2train/gradients/cross_entropy_total/Mean_grad/Cast"/device:GPU:0*#
_output_shapes
:���������*
T0
�
8train/gradients/cross_entropy_total/Reshape_2_grad/ShapeShape1cross_entropy_total/SoftmaxCrossEntropyWithLogits)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
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
ExpandDims:train/gradients/cross_entropy_total/Reshape_2_grad/ReshapeUtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim"/device:GPU:0*'
_output_shapes
:���������*

Tdim0*
T0
�
Jtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/mulMulQtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/ExpandDims3cross_entropy_total/SoftmaxCrossEntropyWithLogits:1"/device:GPU:0*0
_output_shapes
:������������������*
T0
�
6train/gradients/cross_entropy_total/Reshape_grad/ShapeShapeReadout/predicted_labels)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
8train/gradients/cross_entropy_total/Reshape_grad/ReshapeReshapeJtrain/gradients/cross_entropy_total/SoftmaxCrossEntropyWithLogits_grad/mul6train/gradients/cross_entropy_total/Reshape_grad/Shape"/device:GPU:0*
T0*
Tshape0*'
_output_shapes
:���������
�
3train/gradients/Readout/predicted_labels_grad/ShapeShapeReadout/MatMul)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
5train/gradients/Readout/predicted_labels_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
Ctrain/gradients/Readout/predicted_labels_grad/BroadcastGradientArgsBroadcastGradientArgs3train/gradients/Readout/predicted_labels_grad/Shape5train/gradients/Readout/predicted_labels_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
1train/gradients/Readout/predicted_labels_grad/SumSum8train/gradients/cross_entropy_total/Reshape_grad/ReshapeCtrain/gradients/Readout/predicted_labels_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
5train/gradients/Readout/predicted_labels_grad/ReshapeReshape1train/gradients/Readout/predicted_labels_grad/Sum3train/gradients/Readout/predicted_labels_grad/Shape"/device:GPU:0*
T0*
Tshape0*'
_output_shapes
:���������
�
3train/gradients/Readout/predicted_labels_grad/Sum_1Sum8train/gradients/cross_entropy_total/Reshape_grad/ReshapeEtrain/gradients/Readout/predicted_labels_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
7train/gradients/Readout/predicted_labels_grad/Reshape_1Reshape3train/gradients/Readout/predicted_labels_grad/Sum_15train/gradients/Readout/predicted_labels_grad/Shape_1"/device:GPU:0*
Tshape0*
_output_shapes
:*
T0
�
>train/gradients/Readout/predicted_labels_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_16^train/gradients/Readout/predicted_labels_grad/Reshape8^train/gradients/Readout/predicted_labels_grad/Reshape_1"/device:GPU:0
�
Ftrain/gradients/Readout/predicted_labels_grad/tuple/control_dependencyIdentity5train/gradients/Readout/predicted_labels_grad/Reshape?^train/gradients/Readout/predicted_labels_grad/tuple/group_deps"/device:GPU:0*
T0*H
_class>
<:loc:@train/gradients/Readout/predicted_labels_grad/Reshape*'
_output_shapes
:���������
�
Htrain/gradients/Readout/predicted_labels_grad/tuple/control_dependency_1Identity7train/gradients/Readout/predicted_labels_grad/Reshape_1?^train/gradients/Readout/predicted_labels_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:*
T0*J
_class@
><loc:@train/gradients/Readout/predicted_labels_grad/Reshape_1
�
*train/gradients/Readout/MatMul_grad/MatMulMatMulFtrain/gradients/Readout/predicted_labels_grad/tuple/control_dependencyReadout/Variable/read"/device:GPU:0*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
,train/gradients/Readout/MatMul_grad/MatMul_1MatMul'FC1/batch_normalization/batchnorm/add_1Ftrain/gradients/Readout/predicted_labels_grad/tuple/control_dependency"/device:GPU:0*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
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
>train/gradients/Readout/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/Readout/MatMul_grad/MatMul_15^train/gradients/Readout/MatMul_grad/tuple/group_deps"/device:GPU:0*?
_class5
31loc:@train/gradients/Readout/MatMul_grad/MatMul_1*
_output_shapes
:	�*
T0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ShapeShape'FC1/batch_normalization/batchnorm/mul_1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:�*
dtype0
�
Rtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsBtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ShapeDtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape_1"/device:GPU:0*2
_output_shapes 
:���������:���������*
T0
�
@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/SumSum<train/gradients/Readout/MatMul_grad/tuple/control_dependencyRtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/ReshapeReshape@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/SumBtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape"/device:GPU:0*(
_output_shapes
:����������*
T0*
Tshape0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Sum_1Sum<train/gradients/Readout/MatMul_grad/tuple/control_dependencyTtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Ftrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeBtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Sum_1Dtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
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
Wtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityFtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape_1N^train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/group_deps"/device:GPU:0*
_output_shapes	
:�*
T0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/add_1_grad/Reshape_1
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
@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/SumSum@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mulRtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/ReshapeReshape@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/SumBtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Shape"/device:GPU:0*
Tshape0*(
_output_shapes
:����������*
T0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mul_1MulFC1/ReluUtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency"/device:GPU:0*(
_output_shapes
:����������*
T0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Sum_1SumBtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/mul_1Ttrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:�*
dtype0
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
>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/SumSumWtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Ptrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
Btrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/ReshapeReshape>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Sum@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Shape"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Sum_1SumWtrain/gradients/FC1/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Rtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/NegNeg@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Sum_1"/device:GPU:0*
_output_shapes
:*
T0
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1Reshape>train/gradients/FC1/batch_normalization/batchnorm/sub_grad/NegBtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Shape_1"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
Ktrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/ReshapeE^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1"/device:GPU:0
�
Strain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityBtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/ReshapeL^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/group_deps"/device:GPU:0*
_output_shapes	
:�*
T0*U
_classK
IGloc:@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape
�
Utrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityDtrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1L^train/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/group_deps"/device:GPU:0*
_output_shapes	
:�*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/sub_grad/Reshape_1
�
Btrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:�*
dtype0*
_output_shapes
:
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
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/ReshapeReshape@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/SumBtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Shape"/device:GPU:0*
Tshape0*
_output_shapes	
:�*
T0
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
Wtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityFtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape_1N^train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps"/device:GPU:0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/Reshape_1*
_output_shapes	
:�*
T0
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
train/gradients/AddNAddNWtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1Wtrain/gradients/FC1/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1"/device:GPU:0*Y
_classO
MKloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�*
T0
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
@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/mul_1Mul'FC1/batch_normalization/batchnorm/Rsqrttrain/gradients/AddN"/device:GPU:0*
T0*
_output_shapes	
:�
�
@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Sum_1Sum@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/mul_1Rtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1Reshape@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Sum_1Btrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Shape_1"/device:GPU:0*
Tshape0*
_output_shapes	
:�*
T0
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
Utrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityDtrain/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1L^train/gradients/FC1/batch_normalization/batchnorm/mul_grad/tuple/group_deps"/device:GPU:0*
T0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_grad/Reshape_1*
_output_shapes	
:�
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
<train/gradients/FC1/batch_normalization/Select_grad/Select_1SelectFC1/batch_normalization/Reshape>train/gradients/FC1/batch_normalization/Select_grad/zeros_like<train/gradients/FC1/batch_normalization/Squeeze_grad/Reshape"/device:GPU:0*
_output_shapes
:	�*
T0
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
=train/gradients/FC1/batch_normalization/ExpandDims_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB:�*
dtype0
�
?train/gradients/FC1/batch_normalization/ExpandDims_grad/ReshapeReshapeLtrain/gradients/FC1/batch_normalization/Select_grad/tuple/control_dependency=train/gradients/FC1/batch_normalization/ExpandDims_grad/Shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
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
Btrain/gradients/FC1/batch_normalization/batchnorm/add_grad/ReshapeReshape>train/gradients/FC1/batch_normalization/batchnorm/add_grad/Sum@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Shape"/device:GPU:0*
_output_shapes	
:�*
T0*
Tshape0
�
@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Sum_1SumFtrain/gradients/FC1/batch_normalization/batchnorm/Rsqrt_grad/RsqrtGradRtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Dtrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Reshape_1Reshape@train/gradients/FC1/batch_normalization/batchnorm/add_grad/Sum_1Btrain/gradients/FC1/batch_normalization/batchnorm/add_grad/Shape_1"/device:GPU:0*
_output_shapes
: *
T0*
Tshape0
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
<train/gradients/FC1/batch_normalization/Squeeze_1_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
�
>train/gradients/FC1/batch_normalization/Squeeze_1_grad/ReshapeReshapeStrain/gradients/FC1/batch_normalization/batchnorm/add_grad/tuple/control_dependency<train/gradients/FC1/batch_normalization/Squeeze_1_grad/Shape"/device:GPU:0*
Tshape0*
_output_shapes
:	�*
T0
�
@train/gradients/FC1/batch_normalization/Select_1_grad/zeros_likeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:	�*
valueB	�*    *
dtype0
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
Ntrain/gradients/FC1/batch_normalization/Select_1_grad/tuple/control_dependencyIdentity<train/gradients/FC1/batch_normalization/Select_1_grad/SelectG^train/gradients/FC1/batch_normalization/Select_1_grad/tuple/group_deps"/device:GPU:0*O
_classE
CAloc:@train/gradients/FC1/batch_normalization/Select_1_grad/Select*
_output_shapes
:	�*
T0
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
Dtrain/gradients/FC1/batch_normalization/moments/Squeeze_1_grad/ShapeConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0
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
Atrain/gradients/FC1/batch_normalization/moments/variance_grad/addAdd:FC1/batch_normalization/moments/variance/reduction_indicesBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Size"/device:GPU:0*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
_output_shapes
:
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
Itrain/gradients/FC1/batch_normalization/moments/variance_grad/range/startConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
value	B : *V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
dtype0
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
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/FillFillEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_1Htrain/gradients/FC1/batch_normalization/moments/variance_grad/Fill/value"/device:GPU:0*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
_output_shapes
:
�
Ktrain/gradients/FC1/batch_normalization/moments/variance_grad/DynamicStitchDynamicStitchCtrain/gradients/FC1/batch_normalization/moments/variance_grad/rangeAtrain/gradients/FC1/batch_normalization/moments/variance_grad/modCtrain/gradients/FC1/batch_normalization/moments/variance_grad/ShapeBtrain/gradients/FC1/batch_normalization/moments/variance_grad/Fill"/device:GPU:0*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
N*#
_output_shapes
:���������
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
Ftrain/gradients/FC1/batch_normalization/moments/variance_grad/floordivFloorDivCtrain/gradients/FC1/batch_normalization/moments/variance_grad/ShapeEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum"/device:GPU:0*
T0*V
_classL
JHloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape*
_output_shapes
:
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/ReshapeReshapeFtrain/gradients/FC1/batch_normalization/moments/Squeeze_1_grad/ReshapeKtrain/gradients/FC1/batch_normalization/moments/variance_grad/DynamicStitch"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/TileTileEtrain/gradients/FC1/batch_normalization/moments/variance_grad/ReshapeFtrain/gradients/FC1/batch_normalization/moments/variance_grad/floordiv"/device:GPU:0*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2Shape1FC1/batch_normalization/moments/SquaredDifference)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
�
Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_3Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
�
Ctrain/gradients/FC1/batch_normalization/moments/variance_grad/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
valueB: *X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
dtype0
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
Dtrain/gradients/FC1/batch_normalization/moments/variance_grad/Prod_1ProdEtrain/gradients/FC1/batch_normalization/moments/variance_grad/Shape_3Etrain/gradients/FC1/batch_normalization/moments/variance_grad/Const_1"/device:GPU:0*
T0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Itrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1/yConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
value	B :*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
dtype0*
_output_shapes
: 
�
Gtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1MaximumDtrain/gradients/FC1/batch_normalization/moments/variance_grad/Prod_1Itrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1/y"/device:GPU:0*
T0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: 
�
Htrain/gradients/FC1/batch_normalization/moments/variance_grad/floordiv_1FloorDivBtrain/gradients/FC1/batch_normalization/moments/variance_grad/ProdGtrain/gradients/FC1/batch_normalization/moments/variance_grad/Maximum_1"/device:GPU:0*
T0*X
_classN
LJloc:@train/gradients/FC1/batch_normalization/moments/variance_grad/Shape_2*
_output_shapes
: 
�
Btrain/gradients/FC1/batch_normalization/moments/variance_grad/CastCastHtrain/gradients/FC1/batch_normalization/moments/variance_grad/floordiv_1"/device:GPU:0*
_output_shapes
: *

DstT0*

SrcT0
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
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mulMulMtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/scalarEtrain/gradients/FC1/batch_normalization/moments/variance_grad/truediv"/device:GPU:0*(
_output_shapes
:����������*
T0
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/subSubFC1/Relu,FC1/batch_normalization/moments/StopGradient)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1F^train/gradients/FC1/batch_normalization/moments/variance_grad/truediv"/device:GPU:0*(
_output_shapes
:����������*
T0
�
Ltrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mul_1MulJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mulJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/sub"/device:GPU:0*(
_output_shapes
:����������*
T0
�
Jtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/SumSumLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mul_1\train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgs"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Ntrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/ReshapeReshapeJtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/SumLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Shape"/device:GPU:0*
Tshape0*(
_output_shapes
:����������*
T0
�
Ltrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Sum_1SumLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/mul_1^train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/BroadcastGradientArgs:1"/device:GPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
Ptrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Reshape_1ReshapeLtrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Sum_1Ntrain/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/Shape_1"/device:GPU:0*
_output_shapes
:	�*
T0*
Tshape0
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
=train/gradients/FC1/batch_normalization/moments/mean_grad/addAdd6FC1/batch_normalization/moments/mean/reduction_indices>train/gradients/FC1/batch_normalization/moments/mean_grad/Size"/device:GPU:0*
_output_shapes
:*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape
�
=train/gradients/FC1/batch_normalization/moments/mean_grad/modFloorMod=train/gradients/FC1/batch_normalization/moments/mean_grad/add>train/gradients/FC1/batch_normalization/moments/mean_grad/Size"/device:GPU:0*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
_output_shapes
:
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
Btrain/gradients/FC1/batch_normalization/moments/mean_grad/floordivFloorDiv?train/gradients/FC1/batch_normalization/moments/mean_grad/ShapeAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Maximum"/device:GPU:0*
T0*R
_classH
FDloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape*
_output_shapes
:
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/ReshapeReshapeDtrain/gradients/FC1/batch_normalization/moments/Squeeze_grad/ReshapeGtrain/gradients/FC1/batch_normalization/moments/mean_grad/DynamicStitch"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:
�
>train/gradients/FC1/batch_normalization/moments/mean_grad/TileTileAtrain/gradients/FC1/batch_normalization/moments/mean_grad/ReshapeBtrain/gradients/FC1/batch_normalization/moments/mean_grad/floordiv"/device:GPU:0*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2ShapeFC1/Relu)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
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
>train/gradients/FC1/batch_normalization/moments/mean_grad/ProdProdAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2?train/gradients/FC1/batch_normalization/moments/mean_grad/Const"/device:GPU:0*
T0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Const_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB: *T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
dtype0*
_output_shapes
:
�
@train/gradients/FC1/batch_normalization/moments/mean_grad/Prod_1ProdAtrain/gradients/FC1/batch_normalization/moments/mean_grad/Shape_3Atrain/gradients/FC1/batch_normalization/moments/mean_grad/Const_1"/device:GPU:0*
T0*T
_classJ
HFloc:@train/gradients/FC1/batch_normalization/moments/mean_grad/Shape_2*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
>train/gradients/FC1/batch_normalization/moments/mean_grad/CastCastDtrain/gradients/FC1/batch_normalization/moments/mean_grad/floordiv_1"/device:GPU:0*

SrcT0*
_output_shapes
: *

DstT0
�
Atrain/gradients/FC1/batch_normalization/moments/mean_grad/truedivRealDiv>train/gradients/FC1/batch_normalization/moments/mean_grad/Tile>train/gradients/FC1/batch_normalization/moments/mean_grad/Cast"/device:GPU:0*(
_output_shapes
:����������*
T0
�
train/gradients/AddN_1AddNUtrain/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_train/gradients/FC1/batch_normalization/moments/SquaredDifference_grad/tuple/control_dependencyAtrain/gradients/FC1/batch_normalization/moments/mean_grad/truediv"/device:GPU:0*W
_classM
KIloc:@train/gradients/FC1/batch_normalization/batchnorm/mul_1_grad/Reshape*
N*(
_output_shapes
:����������*
T0
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
$train/gradients/FC1/add_grad/ReshapeReshape train/gradients/FC1/add_grad/Sum"train/gradients/FC1/add_grad/Shape"/device:GPU:0*(
_output_shapes
:����������*
T0*
Tshape0
�
"train/gradients/FC1/add_grad/Sum_1Sum&train/gradients/FC1/Relu_grad/ReluGrad4train/gradients/FC1/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
&train/gradients/FC1/add_grad/Reshape_1Reshape"train/gradients/FC1/add_grad/Sum_1$train/gradients/FC1/add_grad/Shape_1"/device:GPU:0*
T0*
Tshape0*
_output_shapes	
:�
�
-train/gradients/FC1/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1%^train/gradients/FC1/add_grad/Reshape'^train/gradients/FC1/add_grad/Reshape_1"/device:GPU:0
�
5train/gradients/FC1/add_grad/tuple/control_dependencyIdentity$train/gradients/FC1/add_grad/Reshape.^train/gradients/FC1/add_grad/tuple/group_deps"/device:GPU:0*(
_output_shapes
:����������*
T0*7
_class-
+)loc:@train/gradients/FC1/add_grad/Reshape
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
(train/gradients/FC1/MatMul_grad/MatMul_1MatMulFC1/Reshape5train/gradients/FC1/add_grad/tuple/control_dependency"/device:GPU:0*
T0*!
_output_shapes
:���*
transpose_a(*
transpose_b( 
�
0train/gradients/FC1/MatMul_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1'^train/gradients/FC1/MatMul_grad/MatMul)^train/gradients/FC1/MatMul_grad/MatMul_1"/device:GPU:0
�
8train/gradients/FC1/MatMul_grad/tuple/control_dependencyIdentity&train/gradients/FC1/MatMul_grad/MatMul1^train/gradients/FC1/MatMul_grad/tuple/group_deps"/device:GPU:0*)
_output_shapes
:�����������*
T0*9
_class/
-+loc:@train/gradients/FC1/MatMul_grad/MatMul
�
:train/gradients/FC1/MatMul_grad/tuple/control_dependency_1Identity(train/gradients/FC1/MatMul_grad/MatMul_11^train/gradients/FC1/MatMul_grad/tuple/group_deps"/device:GPU:0*
T0*;
_class1
/-loc:@train/gradients/FC1/MatMul_grad/MatMul_1*!
_output_shapes
:���
�
&train/gradients/FC1/Reshape_grad/ShapeShape#CNN3/batch_normalization/cond/Merge)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
(train/gradients/FC1/Reshape_grad/ReshapeReshape8train/gradients/FC1/MatMul_grad/tuple/control_dependency&train/gradients/FC1/Reshape_grad/Shape"/device:GPU:0*
Tshape0*/
_output_shapes
:���������@*
T0
�
Btrain/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_gradSwitch(train/gradients/FC1/Reshape_grad/Reshape%CNN3/batch_normalization/cond/pred_id"/device:GPU:0*J
_output_shapes8
6:���������@:���������@*
T0*;
_class1
/-loc:@train/gradients/FC1/Reshape_grad/Reshape
�
Itrain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1C^train/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_grad"/device:GPU:0
�
Qtrain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/control_dependencyIdentityBtrain/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_gradJ^train/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*;
_class1
/-loc:@train/gradients/FC1/Reshape_grad/Reshape*/
_output_shapes
:���������@*
T0
�
Strain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/control_dependency_1IdentityDtrain/gradients/CNN3/batch_normalization/cond/Merge_grad/cond_grad:1J^train/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*
T0*;
_class1
/-loc:@train/gradients/FC1/Reshape_grad/Reshape*/
_output_shapes
:���������@
�
train/gradients/zeros_like_1	ZerosLike0CNN3/batch_normalization/cond/FusedBatchNorm_1:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*
T0
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
Rtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*%
valueB"             *
dtype0
�
Mtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose	Transpose5CNN3/batch_normalization/cond/FusedBatchNorm_1/SwitchRtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/perm"/device:GPU:0*
T0*/
_output_shapes
:���������@*
Tperm0
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
:*%
valueB"             *
dtype0
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
Otrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2	TransposeVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2/perm"/device:GPU:0*/
_output_shapes
:���������@*
Tperm0*
T0
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1W^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradP^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2"/device:GPU:0
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityOtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*b
_classX
VTloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2*/
_output_shapes
:���������@*
T0
�
^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityXtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:
�
^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityXtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:*
T0
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
train/gradients/zeros_like_8	ZerosLike.CNN3/batch_normalization/cond/FusedBatchNorm:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradStrain/gradients/CNN3/batch_normalization/cond/Merge_grad/tuple/control_dependency_15CNN3/batch_normalization/cond/FusedBatchNorm/Switch:17CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1:1.CNN3/batch_normalization/cond/FusedBatchNorm:3.CNN3/batch_normalization/cond/FusedBatchNorm:4"/device:GPU:0*
epsilon%o�:*
T0*
data_formatNCHW*C
_output_shapes1
/:���������@::: : *
is_training(
�
Rtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1U^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad"/device:GPU:0
�
Ztrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradS^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������@
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
:*
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4S^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
: *
T0*g
_class]
[Yloc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
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
train/gradients/Switch_1Switch batch_normalization_2/gamma/read%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
::*
T0
�
train/gradients/Shape_2Shapetrain/gradients/Switch_1:1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_1/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_1Filltrain/gradients/Shape_2train/gradients/zeros_1/Const"/device:GPU:0*
T0*
_output_shapes
:
�
Vtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMerge^train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1train/gradients/zeros_1"/device:GPU:0*
_output_shapes

:: *
T0*
N
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
train/gradients/Switch_3Switch	CNN3/Relu%CNN3/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*J
_output_shapes8
6:���������@:���������@
~
train/gradients/Shape_4Shapetrain/gradients/Switch_3"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
�
train/gradients/zeros_3/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_3Filltrain/gradients/Shape_4train/gradients/zeros_3/Const"/device:GPU:0*/
_output_shapes
:���������@*
T0
�
Rtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_grad/cond_gradMergeZtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencytrain/gradients/zeros_3"/device:GPU:0*1
_output_shapes
:���������@: *
T0*
N
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
train/gradients/zeros_4Filltrain/gradients/Shape_5train/gradients/zeros_4/Const"/device:GPU:0*
_output_shapes
:*
T0
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
train/gradients/Shape_6Shapetrain/gradients/Switch_5"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_5/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_5Filltrain/gradients/Shape_6train/gradients/zeros_5/Const"/device:GPU:0*
T0*
_output_shapes
:
�
Ttrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_gradMerge\train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2train/gradients/zeros_5"/device:GPU:0*
T0*
N*
_output_shapes

:: 
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
train/gradients/AddN_3AddNVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:
�
train/gradients/AddN_4AddNVtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradTtrain/gradients/CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad"/device:GPU:0*
_output_shapes
:*
T0*i
_class_
][loc:@train/gradients/CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N
�
#train/gradients/CNN3/add_grad/ShapeShapeCNN3/Conv2D)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
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
!train/gradients/CNN3/add_grad/SumSum'train/gradients/CNN3/Relu_grad/ReluGrad3train/gradients/CNN3/add_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
%train/gradients/CNN3/add_grad/ReshapeReshape!train/gradients/CNN3/add_grad/Sum#train/gradients/CNN3/add_grad/Shape"/device:GPU:0*/
_output_shapes
:���������@*
T0*
Tshape0
�
#train/gradients/CNN3/add_grad/Sum_1Sum'train/gradients/CNN3/Relu_grad/ReluGrad5train/gradients/CNN3/add_grad/BroadcastGradientArgs:1"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
'train/gradients/CNN3/add_grad/Reshape_1Reshape#train/gradients/CNN3/add_grad/Sum_1%train/gradients/CNN3/add_grad/Shape_1"/device:GPU:0*
_output_shapes
:@*
T0*
Tshape0
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
8train/gradients/CNN3/add_grad/tuple/control_dependency_1Identity'train/gradients/CNN3/add_grad/Reshape_1/^train/gradients/CNN3/add_grad/tuple/group_deps"/device:GPU:0*
T0*:
_class0
.,loc:@train/gradients/CNN3/add_grad/Reshape_1*
_output_shapes
:@
�
'train/gradients/CNN3/Conv2D_grad/ShapeNShapeN#CNN2/batch_normalization/cond/MergeCNN3/weights/Variable/read)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
N* 
_output_shapes
::
�
4train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'train/gradients/CNN3/Conv2D_grad/ShapeNCNN3/weights/Variable/read6train/gradients/CNN3/add_grad/tuple/control_dependency"/device:GPU:0*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������
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
Btrain/gradients/CNN2/batch_normalization/cond/Merge_grad/cond_gradSwitch9train/gradients/CNN3/Conv2D_grad/tuple/control_dependency%CNN2/batch_normalization/cond/pred_id"/device:GPU:0*J
_output_shapes8
6:���������   :���������   *
T0*G
_class=
;9loc:@train/gradients/CNN3/Conv2D_grad/Conv2DBackpropInput
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
train/gradients/zeros_like_11	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:3)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
T0
�
train/gradients/zeros_like_12	ZerosLike0CNN2/batch_normalization/cond/FusedBatchNorm_1:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
T0
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
Vtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradOtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1Mtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose7CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*
epsilon%o�:*
T0*
data_formatNHWC*G
_output_shapes5
3:���������   : : : : *
is_training( 
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
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityOtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������   *
T0*b
_classX
VTloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2
�
^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityXtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityXtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
: *
T0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad
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
train/gradients/zeros_like_16	ZerosLike.CNN2/batch_normalization/cond/FusedBatchNorm:4)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
T0
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradStrain/gradients/CNN2/batch_normalization/cond/Merge_grad/tuple/control_dependency_15CNN2/batch_normalization/cond/FusedBatchNorm/Switch:17CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1:1.CNN2/batch_normalization/cond/FusedBatchNorm:3.CNN2/batch_normalization/cond/FusedBatchNorm:4"/device:GPU:0*
epsilon%o�:*
T0*
data_formatNCHW*C
_output_shapes1
/:���������   : : : : *
is_training(
�
Rtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1U^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad"/device:GPU:0
�
Ztrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradS^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������   *
T0
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: *
T0
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
�
\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4S^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
: *
T0*g
_class]
[Yloc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
train/gradients/Switch_6Switch	CNN2/Relu%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*J
_output_shapes8
6:���������   :���������   
�
train/gradients/Shape_7Shapetrain/gradients/Switch_6:1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
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
T0*
N*1
_output_shapes
:���������   : 
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
train/gradients/zeros_7Filltrain/gradients/Shape_8train/gradients/zeros_7/Const"/device:GPU:0*
_output_shapes
: *
T0
�
Vtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMerge^train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1train/gradients/zeros_7"/device:GPU:0*
_output_shapes

: : *
T0*
N
�
train/gradients/Switch_8Switchbatch_normalization_1/beta/read%CNN2/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
: : 
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
train/gradients/Shape_10Shapetrain/gradients/Switch_9"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
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
N*1
_output_shapes
:���������   : *
T0
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
train/gradients/zeros_10/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_10Filltrain/gradients/Shape_11train/gradients/zeros_10/Const"/device:GPU:0*
_output_shapes
: *
T0
�
Ttrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMerge\train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1train/gradients/zeros_10"/device:GPU:0*
N*
_output_shapes

: : *
T0
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
train/gradients/AddN_7AddNVtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_gradTtrain/gradients/CNN2/batch_normalization/cond/FusedBatchNorm/Switch_2_grad/cond_grad"/device:GPU:0*
_output_shapes
: *
T0*i
_class_
][loc:@train/gradients/CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2_grad/cond_grad*
N
�
#train/gradients/CNN2/add_grad/ShapeShapeCNN2/Conv2D)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
_output_shapes
:*
T0
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
!train/gradients/CNN2/add_grad/SumSum'train/gradients/CNN2/Relu_grad/ReluGrad3train/gradients/CNN2/add_grad/BroadcastGradientArgs"/device:GPU:0*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
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
6train/gradients/CNN2/add_grad/tuple/control_dependencyIdentity%train/gradients/CNN2/add_grad/Reshape/^train/gradients/CNN2/add_grad/tuple/group_deps"/device:GPU:0*
T0*8
_class.
,*loc:@train/gradients/CNN2/add_grad/Reshape*/
_output_shapes
:���������   
�
8train/gradients/CNN2/add_grad/tuple/control_dependency_1Identity'train/gradients/CNN2/add_grad/Reshape_1/^train/gradients/CNN2/add_grad/tuple/group_deps"/device:GPU:0*:
_class0
.,loc:@train/gradients/CNN2/add_grad/Reshape_1*
_output_shapes
: *
T0
�
'train/gradients/CNN2/Conv2D_grad/ShapeNShapeN#CNN1/batch_normalization/cond/MergeCNN2/weights/Variable/read)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
out_type0*
N* 
_output_shapes
::*
T0
�
4train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput'train/gradients/CNN2/Conv2D_grad/ShapeNCNN2/weights/Variable/read6train/gradients/CNN2/add_grad/tuple/control_dependency"/device:GPU:0*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC
�
5train/gradients/CNN2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter#CNN1/batch_normalization/cond/Merge)train/gradients/CNN2/Conv2D_grad/ShapeN:16train/gradients/CNN2/add_grad/tuple/control_dependency"/device:GPU:0*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
Strain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/control_dependency_1IdentityDtrain/gradients/CNN1/batch_normalization/cond/Merge_grad/cond_grad:1J^train/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/group_deps"/device:GPU:0*/
_output_shapes
:���������@@*
T0*G
_class=
;9loc:@train/gradients/CNN2/Conv2D_grad/Conv2DBackpropInput
�
train/gradients/zeros_like_17	ZerosLike0CNN1/batch_normalization/cond/FusedBatchNorm_1:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:@
�
train/gradients/zeros_like_18	ZerosLike0CNN1/batch_normalization/cond/FusedBatchNorm_1:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:@
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
Mtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose	Transpose5CNN1/batch_normalization/cond/FusedBatchNorm_1/SwitchRtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose/perm"/device:GPU:0*/
_output_shapes
:���������@@*
Tperm0*
T0
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/permConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
�
Otrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1	TransposeQtrain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/control_dependencyTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1/perm"/device:GPU:0*/
_output_shapes
:���������@@*
Tperm0*
T0
�
Vtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGradFusedBatchNormGradOtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_1Mtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose7CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_17CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_37CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4"/device:GPU:0*G
_output_shapes5
3:���������@@:@:@:@:@*
is_training( *
epsilon%o�:*
T0*
data_formatNHWC
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
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependencyIdentityOtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2U^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*b
_classX
VTloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/transpose_2*/
_output_shapes
:���������@@*
T0
�
^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1IdentityXtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:1U^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*i
_class_
][loc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:@*
T0
�
^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_2IdentityXtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad:2U^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/group_deps"/device:GPU:0*i
_class_
][loc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/FusedBatchNormGrad*
_output_shapes
:@*
T0
�
train/gradients/zeros_like_21	ZerosLike.CNN1/batch_normalization/cond/FusedBatchNorm:1)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:@
�
train/gradients/zeros_like_22	ZerosLike.CNN1/batch_normalization/cond/FusedBatchNorm:2)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
_output_shapes
:@
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
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradFusedBatchNormGradStrain/gradients/CNN1/batch_normalization/cond/Merge_grad/tuple/control_dependency_15CNN1/batch_normalization/cond/FusedBatchNorm/Switch:17CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1:1.CNN1/batch_normalization/cond/FusedBatchNorm:3.CNN1/batch_normalization/cond/FusedBatchNorm:4"/device:GPU:0*
data_formatNCHW*C
_output_shapes1
/:���������@@:@:@: : *
is_training(*
epsilon%o�:*
T0
�
Rtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1U^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad"/device:GPU:0
�
Ztrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependencyIdentityTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGradS^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*/
_output_shapes
:���������@@*
T0
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:1S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_2IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:2S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
:@
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_3IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:3S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
_output_shapes
: *
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad
�
\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_4IdentityVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad:4S^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/group_deps"/device:GPU:0*
T0*g
_class]
[Yloc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/FusedBatchNormGrad*
_output_shapes
: 
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
T0*
N*1
_output_shapes
:���������@@: 
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
train/gradients/zeros_13/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_13Filltrain/gradients/Shape_14train/gradients/zeros_13/Const"/device:GPU:0*
_output_shapes
:@*
T0
�
Vtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradMerge^train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1_grad/tuple/control_dependency_1train/gradients/zeros_13"/device:GPU:0*
T0*
N*
_output_shapes

:@: 
�
train/gradients/Switch_14Switchbatch_normalization/beta/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
:@:@
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
T0*
N*
_output_shapes

:@: 
�
train/gradients/Switch_15Switch	CNN1/Relu%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*J
_output_shapes8
6:���������@@:���������@@*
T0
�
train/gradients/Shape_16Shapetrain/gradients/Switch_15"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
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
N*1
_output_shapes
:���������@@: *
T0
�
train/gradients/Switch_16Switchbatch_normalization/gamma/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0* 
_output_shapes
:@:@
�
train/gradients/Shape_17Shapetrain/gradients/Switch_16"/device:GPU:0*
T0*
out_type0*
_output_shapes
:
�
train/gradients/zeros_16/ConstConst)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
_output_shapes
: *
valueB
 *    *
dtype0
�
train/gradients/zeros_16Filltrain/gradients/Shape_17train/gradients/zeros_16/Const"/device:GPU:0*
T0*
_output_shapes
:@
�
Ttrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_gradMerge\train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_grad/tuple/control_dependency_1train/gradients/zeros_16"/device:GPU:0*
T0*
N*
_output_shapes

:@: 
�
train/gradients/Switch_17Switchbatch_normalization/beta/read%CNN1/batch_normalization/cond/pred_id)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0* 
_output_shapes
:@:@*
T0
�
train/gradients/Shape_18Shapetrain/gradients/Switch_17"/device:GPU:0*
_output_shapes
:*
T0*
out_type0
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
'train/gradients/CNN1/Relu_grad/ReluGradReluGradtrain/gradients/AddN_8	CNN1/Relu"/device:GPU:0*
T0*/
_output_shapes
:���������@@
�
train/gradients/AddN_9AddNVtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_gradTtrain/gradients/CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1_grad/cond_grad"/device:GPU:0*
T0*i
_class_
][loc:@train/gradients/CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1_grad/cond_grad*
N*
_output_shapes
:@
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
%train/gradients/CNN1/add_grad/Shape_1Const)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
valueB:*
dtype0*
_output_shapes
:
�
3train/gradients/CNN1/add_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/CNN1/add_grad/Shape%train/gradients/CNN1/add_grad/Shape_1"/device:GPU:0*
T0*2
_output_shapes 
:���������:���������
�
!train/gradients/CNN1/add_grad/SumSum'train/gradients/CNN1/Relu_grad/ReluGrad3train/gradients/CNN1/add_grad/BroadcastGradientArgs"/device:GPU:0*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
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
'train/gradients/CNN1/add_grad/Reshape_1Reshape#train/gradients/CNN1/add_grad/Sum_1%train/gradients/CNN1/add_grad/Shape_1"/device:GPU:0*
_output_shapes
:*
T0*
Tshape0
�
.train/gradients/CNN1/add_grad/tuple/group_depsNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1&^train/gradients/CNN1/add_grad/Reshape(^train/gradients/CNN1/add_grad/Reshape_1"/device:GPU:0
�
6train/gradients/CNN1/add_grad/tuple/control_dependencyIdentity%train/gradients/CNN1/add_grad/Reshape/^train/gradients/CNN1/add_grad/tuple/group_deps"/device:GPU:0*8
_class.
,*loc:@train/gradients/CNN1/add_grad/Reshape*/
_output_shapes
:���������@@*
T0
�
8train/gradients/CNN1/add_grad/tuple/control_dependency_1Identity'train/gradients/CNN1/add_grad/Reshape_1/^train/gradients/CNN1/add_grad/tuple/group_deps"/device:GPU:0*
T0*:
_class0
.,loc:@train/gradients/CNN1/add_grad/Reshape_1*
_output_shapes
:
�
'train/gradients/CNN1/Conv2D_grad/ShapeNShapeNinput/imagesCNN1/weights/Variable/read)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_1"/device:GPU:0*
T0*
out_type0*
N* 
_output_shapes
::
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
5train/gradients/CNN1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput/images)train/gradients/CNN1/Conv2D_grad/ShapeN:16train/gradients/CNN1/add_grad/tuple/control_dependency"/device:GPU:0*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4������������������������������������*
T0*
strides
*
data_formatNHWC
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
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
: 
�
train/beta1_power/readIdentitytrain/beta1_power"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
: *
T0
�
train/beta2_power/initial_valueConst"/device:GPU:0*
valueB
 *w�?*'
_class
loc:@CNN1/biases/Variable*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2"/device:GPU:0*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@CNN1/biases/Variable
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value"/device:GPU:0*
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
: 
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
VariableV2"/device:GPU:0*
shape:*
dtype0*&
_output_shapes
:*
shared_name *(
_class
loc:@CNN1/weights/Variable*
	container 
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
CNN1/weights/Variable/Adam/readIdentityCNN1/weights/Variable/Adam"/device:GPU:0*
T0*(
_class
loc:@CNN1/weights/Variable*&
_output_shapes
:
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
VariableV2"/device:GPU:0*
shared_name *(
_class
loc:@CNN1/weights/Variable*
	container *
shape:*
dtype0*&
_output_shapes
:
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
+CNN1/biases/Variable/Adam/Initializer/zerosConst"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
valueB*    *
dtype0*
_output_shapes
:
�
CNN1/biases/Variable/Adam
VariableV2"/device:GPU:0*
_output_shapes
:*
shared_name *'
_class
loc:@CNN1/biases/Variable*
	container *
shape:*
dtype0
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
VariableV2"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
"CNN1/biases/Variable/Adam_1/AssignAssignCNN1/biases/Variable/Adam_1-CNN1/biases/Variable/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(
�
 CNN1/biases/Variable/Adam_1/readIdentityCNN1/biases/Variable/Adam_1"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
:*
T0
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
%batch_normalization/gamma/Adam/AssignAssignbatch_normalization/gamma/Adam0batch_normalization/gamma/Adam/Initializer/zeros"/device:GPU:0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
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
'batch_normalization/gamma/Adam_1/AssignAssign batch_normalization/gamma/Adam_12batch_normalization/gamma/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(*
_output_shapes
:@
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
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@batch_normalization/beta
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
"batch_normalization/beta/Adam/readIdentitybatch_normalization/beta/Adam"/device:GPU:0*
_output_shapes
:@*
T0*+
_class!
loc:@batch_normalization/beta
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
&batch_normalization/beta/Adam_1/AssignAssignbatch_normalization/beta/Adam_11batch_normalization/beta/Adam_1/Initializer/zeros"/device:GPU:0*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@batch_normalization/beta*
validate_shape(
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
VariableV2"/device:GPU:0*
shared_name *(
_class
loc:@CNN2/weights/Variable*
	container *
shape: *
dtype0*&
_output_shapes
: 
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
#CNN2/weights/Variable/Adam_1/AssignAssignCNN2/weights/Variable/Adam_1.CNN2/weights/Variable/Adam_1/Initializer/zeros"/device:GPU:0*(
_class
loc:@CNN2/weights/Variable*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0
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
shape: *
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@CNN2/biases/Variable*
	container 
�
 CNN2/biases/Variable/Adam/AssignAssignCNN2/biases/Variable/Adam+CNN2/biases/Variable/Adam/Initializer/zeros"/device:GPU:0*'
_class
loc:@CNN2/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
CNN2/biases/Variable/Adam/readIdentityCNN2/biases/Variable/Adam"/device:GPU:0*
T0*'
_class
loc:@CNN2/biases/Variable*
_output_shapes
: 
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
VariableV2"/device:GPU:0*'
_class
loc:@CNN2/biases/Variable*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
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
 CNN2/biases/Variable/Adam_1/readIdentityCNN2/biases/Variable/Adam_1"/device:GPU:0*
T0*'
_class
loc:@CNN2/biases/Variable*
_output_shapes
: 
�
2batch_normalization_1/gamma/Adam/Initializer/zerosConst"/device:GPU:0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_1/gamma*
valueB *    *
dtype0
�
 batch_normalization_1/gamma/Adam
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
VariableV2"/device:GPU:0*
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@batch_normalization_1/gamma*
	container *
shape: 
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
VariableV2"/device:GPU:0*
shape: *
dtype0*
_output_shapes
: *
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container 
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
$batch_normalization_1/beta/Adam/readIdentitybatch_normalization_1/beta/Adam"/device:GPU:0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: *
T0
�
3batch_normalization_1/beta/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_1/beta*
valueB *    *
dtype0
�
!batch_normalization_1/beta/Adam_1
VariableV2"/device:GPU:0*
shape: *
dtype0*
_output_shapes
: *
shared_name *-
_class#
!loc:@batch_normalization_1/beta*
	container 
�
(batch_normalization_1/beta/Adam_1/AssignAssign!batch_normalization_1/beta/Adam_13batch_normalization_1/beta/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_1/beta*
validate_shape(*
_output_shapes
: 
�
&batch_normalization_1/beta/Adam_1/readIdentity!batch_normalization_1/beta/Adam_1"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: 
�
,CNN3/weights/Variable/Adam/Initializer/zerosConst"/device:GPU:0*&
_output_shapes
: @*(
_class
loc:@CNN3/weights/Variable*%
valueB @*    *
dtype0
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
#CNN3/weights/Variable/Adam_1/AssignAssignCNN3/weights/Variable/Adam_1.CNN3/weights/Variable/Adam_1/Initializer/zeros"/device:GPU:0*(
_class
loc:@CNN3/weights/Variable*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0
�
!CNN3/weights/Variable/Adam_1/readIdentityCNN3/weights/Variable/Adam_1"/device:GPU:0*
T0*(
_class
loc:@CNN3/weights/Variable*&
_output_shapes
: @
�
+CNN3/biases/Variable/Adam/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:@*'
_class
loc:@CNN3/biases/Variable*
valueB@*    *
dtype0
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
CNN3/biases/Variable/Adam/readIdentityCNN3/biases/Variable/Adam"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
_output_shapes
:@*
T0
�
-CNN3/biases/Variable/Adam_1/Initializer/zerosConst"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
valueB@*    *
dtype0*
_output_shapes
:@
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
"CNN3/biases/Variable/Adam_1/AssignAssignCNN3/biases/Variable/Adam_1-CNN3/biases/Variable/Adam_1/Initializer/zeros"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
 CNN3/biases/Variable/Adam_1/readIdentityCNN3/biases/Variable/Adam_1"/device:GPU:0*'
_class
loc:@CNN3/biases/Variable*
_output_shapes
:@*
T0
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
VariableV2"/device:GPU:0*
shape:*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container 
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
%batch_normalization_2/gamma/Adam/readIdentity batch_normalization_2/gamma/Adam"/device:GPU:0*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
:*
T0
�
4batch_normalization_2/gamma/Adam_1/Initializer/zerosConst"/device:GPU:0*
_output_shapes
:*.
_class$
" loc:@batch_normalization_2/gamma*
valueB*    *
dtype0
�
"batch_normalization_2/gamma/Adam_1
VariableV2"/device:GPU:0*
_output_shapes
:*
shared_name *.
_class$
" loc:@batch_normalization_2/gamma*
	container *
shape:*
dtype0
�
)batch_normalization_2/gamma/Adam_1/AssignAssign"batch_normalization_2/gamma/Adam_14batch_normalization_2/gamma/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*.
_class$
" loc:@batch_normalization_2/gamma*
validate_shape(*
_output_shapes
:
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
&batch_normalization_2/beta/Adam/AssignAssignbatch_normalization_2/beta/Adam1batch_normalization_2/beta/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
:
�
$batch_normalization_2/beta/Adam/readIdentitybatch_normalization_2/beta/Adam"/device:GPU:0*
_output_shapes
:*
T0*-
_class#
!loc:@batch_normalization_2/beta
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
(batch_normalization_2/beta/Adam_1/AssignAssign!batch_normalization_2/beta/Adam_13batch_normalization_2/beta/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_2/beta*
validate_shape(*
_output_shapes
:
�
&batch_normalization_2/beta/Adam_1/readIdentity!batch_normalization_2/beta/Adam_1"/device:GPU:0*
_output_shapes
:*
T0*-
_class#
!loc:@batch_normalization_2/beta
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
dtype0*!
_output_shapes
:���*
shared_name *
_class
loc:@FC1/Variable*
	container *
shape:���
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
FC1/Variable/Adam/readIdentityFC1/Variable/Adam"/device:GPU:0*
T0*
_class
loc:@FC1/Variable*!
_output_shapes
:���
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
FC1/Variable/Adam_1/AssignAssignFC1/Variable/Adam_1%FC1/Variable/Adam_1/Initializer/zeros"/device:GPU:0*!
_output_shapes
:���*
use_locking(*
T0*
_class
loc:@FC1/Variable*
validate_shape(
�
FC1/Variable/Adam_1/readIdentityFC1/Variable/Adam_1"/device:GPU:0*
T0*
_class
loc:@FC1/Variable*!
_output_shapes
:���
�
%FC1/Variable_1/Adam/Initializer/zerosConst"/device:GPU:0*
_output_shapes	
:�*!
_class
loc:@FC1/Variable_1*
valueB�*    *
dtype0
�
FC1/Variable_1/Adam
VariableV2"/device:GPU:0*
dtype0*
_output_shapes	
:�*
shared_name *!
_class
loc:@FC1/Variable_1*
	container *
shape:�
�
FC1/Variable_1/Adam/AssignAssignFC1/Variable_1/Adam%FC1/Variable_1/Adam/Initializer/zeros"/device:GPU:0*!
_class
loc:@FC1/Variable_1*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
VariableV2"/device:GPU:0*
shared_name *!
_class
loc:@FC1/Variable_1*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
FC1/Variable_1/Adam_1/readIdentityFC1/Variable_1/Adam_1"/device:GPU:0*!
_class
loc:@FC1/Variable_1*
_output_shapes	
:�*
T0
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
'batch_normalization_3/gamma/Adam_1/readIdentity"batch_normalization_3/gamma/Adam_1"/device:GPU:0*
T0*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes	
:�
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
VariableV2"/device:GPU:0*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@batch_normalization_3/beta*
	container 
�
&batch_normalization_3/beta/Adam/AssignAssignbatch_normalization_3/beta/Adam1batch_normalization_3/beta/Adam/Initializer/zeros"/device:GPU:0*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(
�
$batch_normalization_3/beta/Adam/readIdentitybatch_normalization_3/beta/Adam"/device:GPU:0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:�*
T0
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
VariableV2"/device:GPU:0*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@batch_normalization_3/beta
�
(batch_normalization_3/beta/Adam_1/AssignAssign!batch_normalization_3/beta/Adam_13batch_normalization_3/beta/Adam_1/Initializer/zeros"/device:GPU:0*-
_class#
!loc:@batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
&batch_normalization_3/beta/Adam_1/readIdentity!batch_normalization_3/beta/Adam_1"/device:GPU:0*
T0*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes	
:�
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
Readout/Variable/Adam/AssignAssignReadout/Variable/Adam'Readout/Variable/Adam/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*#
_class
loc:@Readout/Variable*
validate_shape(*
_output_shapes
:	�
�
Readout/Variable/Adam/readIdentityReadout/Variable/Adam"/device:GPU:0*
T0*#
_class
loc:@Readout/Variable*
_output_shapes
:	�
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
VariableV2"/device:GPU:0*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *%
_class
loc:@Readout/Variable_1
�
Readout/Variable_1/Adam/AssignAssignReadout/Variable_1/Adam)Readout/Variable_1/Adam/Initializer/zeros"/device:GPU:0*
_output_shapes
:*
use_locking(*
T0*%
_class
loc:@Readout/Variable_1*
validate_shape(
�
Readout/Variable_1/Adam/readIdentityReadout/Variable_1/Adam"/device:GPU:0*
T0*%
_class
loc:@Readout/Variable_1*
_output_shapes
:
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
 Readout/Variable_1/Adam_1/AssignAssignReadout/Variable_1/Adam_1+Readout/Variable_1/Adam_1/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*%
_class
loc:@Readout/Variable_1*
validate_shape(*
_output_shapes
:
�
Readout/Variable_1/Adam_1/readIdentityReadout/Variable_1/Adam_1"/device:GPU:0*
_output_shapes
:*
T0*%
_class
loc:@Readout/Variable_1
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
1train/Adam/update_CNN2/weights/Variable/ApplyAdam	ApplyAdamCNN2/weights/VariableCNN2/weights/Variable/AdamCNN2/weights/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/CNN2/Conv2D_grad/tuple/control_dependency_1"/device:GPU:0*(
_class
loc:@CNN2/weights/Variable*
use_nesterov( *&
_output_shapes
: *
use_locking( *
T0
�
0train/Adam/update_CNN2/biases/Variable/ApplyAdam	ApplyAdamCNN2/biases/VariableCNN2/biases/Variable/AdamCNN2/biases/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon8train/gradients/CNN2/add_grad/tuple/control_dependency_1"/device:GPU:0*
use_locking( *
T0*'
_class
loc:@CNN2/biases/Variable*
use_nesterov( *
_output_shapes
: 
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
6train/Adam/update_batch_normalization_2/beta/ApplyAdam	ApplyAdambatch_normalization_2/betabatch_normalization_2/beta/Adam!batch_normalization_2/beta/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilontrain/gradients/AddN_4"/device:GPU:0*-
_class#
!loc:@batch_normalization_2/beta*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
(train/Adam/update_FC1/Variable/ApplyAdam	ApplyAdamFC1/VariableFC1/Variable/AdamFC1/Variable/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/FC1/MatMul_grad/tuple/control_dependency_1"/device:GPU:0*
_class
loc:@FC1/Variable*
use_nesterov( *!
_output_shapes
:���*
use_locking( *
T0
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
6train/Adam/update_batch_normalization_3/beta/ApplyAdam	ApplyAdambatch_normalization_3/betabatch_normalization_3/beta/Adam!batch_normalization_3/beta/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonStrain/gradients/FC1/batch_normalization/batchnorm/sub_grad/tuple/control_dependency"/device:GPU:0*
_output_shapes	
:�*
use_locking( *
T0*-
_class#
!loc:@batch_normalization_3/beta*
use_nesterov( 
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
.train/Adam/update_Readout/Variable_1/ApplyAdam	ApplyAdamReadout/Variable_1Readout/Variable_1/AdamReadout/Variable_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonHtrain/gradients/Readout/predicted_labels_grad/tuple/control_dependency_1"/device:GPU:0*
use_locking( *
T0*%
_class
loc:@Readout/Variable_1*
use_nesterov( *
_output_shapes
:
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta12^train/Adam/update_CNN1/weights/Variable/ApplyAdam1^train/Adam/update_CNN1/biases/Variable/ApplyAdam6^train/Adam/update_batch_normalization/gamma/ApplyAdam5^train/Adam/update_batch_normalization/beta/ApplyAdam2^train/Adam/update_CNN2/weights/Variable/ApplyAdam1^train/Adam/update_CNN2/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_1/gamma/ApplyAdam7^train/Adam/update_batch_normalization_1/beta/ApplyAdam2^train/Adam/update_CNN3/weights/Variable/ApplyAdam1^train/Adam/update_CNN3/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_2/gamma/ApplyAdam7^train/Adam/update_batch_normalization_2/beta/ApplyAdam)^train/Adam/update_FC1/Variable/ApplyAdam+^train/Adam/update_FC1/Variable_1/ApplyAdam8^train/Adam/update_batch_normalization_3/gamma/ApplyAdam7^train/Adam/update_batch_normalization_3/beta/ApplyAdam-^train/Adam/update_Readout/Variable/ApplyAdam/^train/Adam/update_Readout/Variable_1/ApplyAdam"/device:GPU:0*
_output_shapes
: *
T0*'
_class
loc:@CNN1/biases/Variable
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul"/device:GPU:0*
_output_shapes
: *
use_locking( *
T0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta22^train/Adam/update_CNN1/weights/Variable/ApplyAdam1^train/Adam/update_CNN1/biases/Variable/ApplyAdam6^train/Adam/update_batch_normalization/gamma/ApplyAdam5^train/Adam/update_batch_normalization/beta/ApplyAdam2^train/Adam/update_CNN2/weights/Variable/ApplyAdam1^train/Adam/update_CNN2/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_1/gamma/ApplyAdam7^train/Adam/update_batch_normalization_1/beta/ApplyAdam2^train/Adam/update_CNN3/weights/Variable/ApplyAdam1^train/Adam/update_CNN3/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_2/gamma/ApplyAdam7^train/Adam/update_batch_normalization_2/beta/ApplyAdam)^train/Adam/update_FC1/Variable/ApplyAdam+^train/Adam/update_FC1/Variable_1/ApplyAdam8^train/Adam/update_batch_normalization_3/gamma/ApplyAdam7^train/Adam/update_batch_normalization_3/beta/ApplyAdam-^train/Adam/update_Readout/Variable/ApplyAdam/^train/Adam/update_Readout/Variable_1/ApplyAdam"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
_output_shapes
: *
T0
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1"/device:GPU:0*'
_class
loc:@CNN1/biases/Variable*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�


train/AdamNoOp)^CNN1/batch_normalization/AssignMovingAvg+^CNN1/batch_normalization/AssignMovingAvg_1)^CNN2/batch_normalization/AssignMovingAvg+^CNN2/batch_normalization/AssignMovingAvg_1)^CNN3/batch_normalization/AssignMovingAvg+^CNN3/batch_normalization/AssignMovingAvg_1(^FC1/batch_normalization/AssignMovingAvg*^FC1/batch_normalization/AssignMovingAvg_12^train/Adam/update_CNN1/weights/Variable/ApplyAdam1^train/Adam/update_CNN1/biases/Variable/ApplyAdam6^train/Adam/update_batch_normalization/gamma/ApplyAdam5^train/Adam/update_batch_normalization/beta/ApplyAdam2^train/Adam/update_CNN2/weights/Variable/ApplyAdam1^train/Adam/update_CNN2/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_1/gamma/ApplyAdam7^train/Adam/update_batch_normalization_1/beta/ApplyAdam2^train/Adam/update_CNN3/weights/Variable/ApplyAdam1^train/Adam/update_CNN3/biases/Variable/ApplyAdam8^train/Adam/update_batch_normalization_2/gamma/ApplyAdam7^train/Adam/update_batch_normalization_2/beta/ApplyAdam)^train/Adam/update_FC1/Variable/ApplyAdam+^train/Adam/update_FC1/Variable_1/ApplyAdam8^train/Adam/update_batch_normalization_3/gamma/ApplyAdam7^train/Adam/update_batch_normalization_3/beta/ApplyAdam-^train/Adam/update_Readout/Variable/ApplyAdam/^train/Adam/update_Readout/Variable_1/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1"/device:GPU:0
s
"accuracy/correct_results/dimensionConst"/device:GPU:0*
_output_shapes
: *
value	B :*
dtype0
�
accuracy/correct_resultsArgMaxinput/correct_labels"accuracy/correct_results/dimension"/device:GPU:0*#
_output_shapes
:���������*

Tidx0*
T0*
output_type0	
�
accuracy/results_matchEqualaccuracy/correct_resultsReadout/predicted_results"/device:GPU:0*#
_output_shapes
:���������*
T0	
y
accuracy/CastCastaccuracy/results_match"/device:GPU:0*#
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
VariableV2"/device:GPU:0*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *5
_class+
)'loc:@accuracy_streaming_mean/mean/total
�
)accuracy_streaming_mean/mean/total/AssignAssign"accuracy_streaming_mean/mean/total4accuracy_streaming_mean/mean/total/Initializer/zeros"/device:GPU:0*
use_locking(*
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
validate_shape(*
_output_shapes
: 
�
'accuracy_streaming_mean/mean/total/readIdentity"accuracy_streaming_mean/mean/total"/device:GPU:0*
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/total*
_output_shapes
: 
�
4accuracy_streaming_mean/mean/count/Initializer/zerosConst*
_output_shapes
: *5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
valueB
 *    *
dtype0
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
&accuracy_streaming_mean/mean/ToFloat_1Cast!accuracy_streaming_mean/mean/Size"/device:GPU:0*
_output_shapes
: *

DstT0*

SrcT0
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
&accuracy_streaming_mean/mean/AssignAdd	AssignAdd"accuracy_streaming_mean/mean/total accuracy_streaming_mean/mean/Sum"/device:GPU:0*
_output_shapes
: *
use_locking( *
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/total
�
(accuracy_streaming_mean/mean/AssignAdd_1	AssignAdd"accuracy_streaming_mean/mean/count&accuracy_streaming_mean/mean/ToFloat_1^accuracy/accuracy"/device:GPU:0*
use_locking( *
T0*5
_class+
)'loc:@accuracy_streaming_mean/mean/count*
_output_shapes
: 
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
"accuracy_streaming_mean/mean/valueSelect$accuracy_streaming_mean/mean/Greater$accuracy_streaming_mean/mean/truediv$accuracy_streaming_mean/mean/value/e"/device:GPU:0*
_output_shapes
: *
T0
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
(accuracy_streaming_mean/mean/update_op/eConst"/device:GPU:0*
valueB
 *    *
dtype0*
_output_shapes
: 
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
accuracy_streaming_mean_1ScalarSummaryaccuracy_streaming_mean_1/tags"accuracy_streaming_mean/mean/value"/device:GPU:0*
T0*
_output_shapes
: 
�
Merge/MergeSummaryMergeSummaryCNN1/weights/summaries/meanCNN1/weights/summaries/stddev_1CNN1/weights/summaries/maxCNN1/weights/summaries/min CNN1/weights/summaries/histogramCNN1/biases/summaries/meanCNN1/biases/summaries/stddev_1CNN1/biases/summaries/maxCNN1/biases/summaries/minCNN1/biases/summaries/histogramCNN1/activationsCNN1/batch_normCNN2/weights/summaries/meanCNN2/weights/summaries/stddev_1CNN2/weights/summaries/maxCNN2/weights/summaries/min CNN2/weights/summaries/histogramCNN2/biases/summaries/meanCNN2/biases/summaries/stddev_1CNN2/biases/summaries/maxCNN2/biases/summaries/minCNN2/biases/summaries/histogramCNN2/activationsCNN2/batch_normCNN3/weights/summaries/meanCNN3/weights/summaries/stddev_1CNN3/weights/summaries/maxCNN3/weights/summaries/min CNN3/weights/summaries/histogramCNN3/biases/summaries/meanCNN3/biases/summaries/stddev_1CNN3/biases/summaries/maxCNN3/biases/summaries/minCNN3/biases/summaries/histogramCNN3/activationsCNN3/batch_normaccuracy_streaming_mean_1"/device:GPU:0*
_output_shapes
: *
N%""�
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
 batch_normalization/gamma/read:0D
CNN1/Relu:05CNN1/batch_normalization/cond/FusedBatchNorm/Switch:1[
 batch_normalization/gamma/read:07CNN1/batch_normalization/cond/FusedBatchNorm/Switch_1:1Z
batch_normalization/beta/read:07CNN1/batch_normalization/cond/FusedBatchNorm/Switch_2:1
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
*batch_normalization/moving_variance/read:0g
*batch_normalization/moving_variance/read:09CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_4:0c
&batch_normalization/moving_mean/read:09CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_3:0F
CNN1/Relu:07CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch:0]
 batch_normalization/gamma/read:09CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_1:0\
batch_normalization/beta/read:09CNN1/batch_normalization/cond/FusedBatchNorm_1/Switch_2:0
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
,batch_normalization_1/moving_variance/read:0i
,batch_normalization_1/moving_variance/read:09CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_4:0e
(batch_normalization_1/moving_mean/read:09CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_3:0F
CNN2/Relu:07CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch:0_
"batch_normalization_1/gamma/read:09CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_1:0^
!batch_normalization_1/beta/read:09CNN2/batch_normalization/cond/FusedBatchNorm_1/Switch_2:0
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
"batch_normalization_2/gamma/read:0\
!batch_normalization_2/beta/read:07CNN3/batch_normalization/cond/FusedBatchNorm/Switch_2:1D
CNN3/Relu:05CNN3/batch_normalization/cond/FusedBatchNorm/Switch:1]
"batch_normalization_2/gamma/read:07CNN3/batch_normalization/cond/FusedBatchNorm/Switch_1:1
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
,batch_normalization_2/moving_variance/read:0F
CNN3/Relu:07CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch:0_
"batch_normalization_2/gamma/read:09CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_1:0i
,batch_normalization_2/moving_variance/read:09CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_4:0e
(batch_normalization_2/moving_mean/read:09CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_3:0^
!batch_normalization_2/beta/read:09CNN3/batch_normalization/cond/FusedBatchNorm_1/Switch_2:0"�

update_ops�
�
*CNN1/batch_normalization/AssignMovingAvg:0
,CNN1/batch_normalization/AssignMovingAvg_1:0
*CNN2/batch_normalization/AssignMovingAvg:0
,CNN2/batch_normalization/AssignMovingAvg_1:0
*CNN3/batch_normalization/AssignMovingAvg:0
,CNN3/batch_normalization/AssignMovingAvg_1:0
)FC1/batch_normalization/AssignMovingAvg:0
+FC1/batch_normalization/AssignMovingAvg_1:0"�
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
accuracy_streaming_mean_1:0a����      Pɇv	�S��A*��
"
CNN1/weights/summaries/mean�rĻ
&
CNN1/weights/summaries/stddev_1k�=
!
CNN1/weights/summaries/max�X>
!
CNN1/weights/summaries/minp�r�
�
 CNN1/weights/summaries/histogram*�	    NVο    ��?     ��@! ������)u�f}LB#@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�d�\D�X=���%>��:�uܬ�@8�>h�'�?x?�x�?��d�r?�5�i}1?�[^:��"?U�4@@�$?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�T���C?a�$��{E?
����G?�qU���I?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�              @      (@      &@      4@      5@      8@      5@     �E@      4@      :@     �@@     �A@      =@      <@      0@      <@      @@      2@      :@      4@      3@      (@      .@       @       @      @      @      @      @       @       @      @      @      @      @      @      @      @      �?       @      @      @       @      �?      @       @      �?      �?      �?      �?              �?       @              �?       @              �?              �?      �?              �?              �?               @              �?      �?      �?              �?              �?              �?              �?      @       @              �?      �?      �?      �?               @               @      @      @       @      �?      @      @      @      @      @       @      @      (@      @      (@       @      @      "@      &@      .@      8@      1@      4@      ;@      7@      6@      8@      9@      >@      4@     �D@      2@      <@      <@      8@      0@      *@      *@      �?        
!
CNN1/biases/summaries/meanIe�=
%
CNN1/biases/summaries/stddev_1[~�<
 
CNN1/biases/summaries/max5�">
 
CNN1/biases/summaries/min���=
�
CNN1/biases/summaries/histogram*�	   ��r�?   ��P�?      0@!   $��?)�,A�ƒ�?2P�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:P               @      �?      @       @      @              @      �?        
�
CNN1/activations*�    `It@     D�A!��6	BF�A)~� ��SB2�        �-���q=��n����>�u`P+d�>�[�=�k�>��~���>K+�E���>jqs&\��>�ѩ�-�>���%�>�uE����>�f����>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�*ǣ0?@��#G�'A@C\t��B@J23��D@R��'�F@t�n̫I@��`��K@���cN@Π54�P@�0n9�bR@��X�9T@
6@�F?V@r;`�xX@��iI�Z@~
�i�]@��GL:I`@߱�&�a@C��ôc@�Y��=�e@����*�g@�'�;�:j@}+��
�l@�|�6��o@�D:�tq@iK�i3s@�lؔ�u@�������:�           �8śA              �?              �?              �?              @       @      �?              �?      �?      �?      �?       @      @       @      �?      @       @      @      @       @      @      �?      @      @      @      @      @      "@      "@      @      @      $@      (@      (@      *@      &@      &@      ,@      3@      7@      B@      B@      >@      B@      H@     �E@     �A@     �L@     �D@     �H@      S@      L@     @S@     @R@     �U@      [@     �W@     @X@     �^@     �`@      d@     �e@     @g@     �h@     �l@     �m@     �p@     ps@     �u@     �u@     `w@     {@     @~@     ��@     �@     ��@     �@     0�@      �@     8�@      �@     ,�@     ��@     �@     �@     ��@     |�@     8�@     Z�@     ��@      �@     ڧ@     �@     ��@     ��@     Ӱ@     ��@     ��@     ��@     ��@     ��@     5�@    �s�@     �@     �@     8�@    �W�@     ��@     6�@     X�@    @��@    @��@     ��@      �@    �z�@     �@    @��@    ���@    @k�@    �Y�@    ���@    ��@     ��@    ���@    p��@    ���@    Pv�@    ���@    �m�@     H�@    X. A    �A    �A    ((A    �A    h�A    ��A    <3A    �@A    @�A    `2A    A    uA    �%!A    �>#A    ��%A    $�(A    T,A    ��/A    �K2A    �	5A    �68A    �<A    �1@A   �˭BA   ��|EA   ��HA    (�KA   ��OA   �sjQA   ��\SA   @��UA   @��WA    ��WA   �vWA    m�WA   @)�YA   ���ZA   ���ZA   ��OXA   @9�RA   �x�KA    ��HA   ���@A    �S4A    �/A    �<)A    �#A    ��A    ��A    �%A    @�
A    ��A    ���@    �5�@     ��@     �c@        
�0
CNN1/batch_norm*�0	   ��   ��D&@     D�A!�q�	j�hA)���-�A2�������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z���4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�ہkVl�p�w`f���n�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�u 5�9��z��6��`�}6D>��Ő�;F>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>ہkVl�p>BvŐ�r>�H5�8�t>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@�������:�            D�;A   H�A   �x��A    �[A   @=1YA    9{7A    @�6A    ��5A    ��4A    �3A    D�2A    �1A    �0A    ��.A    Z�,A    H�*A    ��(A    ��&A    6�$A    X+#A    :�!A    �' A    t�A    (A    �A    ��A    t�A    ��A    �%A    %A    ��A     �	A    ��A    8�A    �A    H�A    �2 A     ��@    ���@    @b�@    �:�@    �S�@    �`�@    ���@    �n�@    ���@     �@    ��@    ���@    `��@    �Y�@     |�@     ��@    ���@     ��@    �Y�@    ���@    @��@    ���@     M�@     ��@     �@     ��@    ���@    �#�@    �D�@     ��@     ��@     ��@     ��@     
�@     ��@     ܰ@     �@     f�@     ��@     ��@     `�@     R�@     b�@     �@     �@     ,�@     ԗ@     L�@     |�@     ��@     (�@     �@     ��@     x�@     ��@     h�@     H�@     ��@     �@     �y@     `v@     �w@     �u@     �q@      q@      n@     �k@     `l@     `f@     �c@      `@     �b@      ^@      ^@     @[@     �W@     �S@     @Q@     �R@      R@     @P@     �J@      H@      @@      A@      J@      D@      A@      9@      ,@      ;@      4@      ,@      1@      (@      3@      *@      (@      ,@       @      &@      @       @      $@      @      $@      @      "@      @      �?       @      @       @      @      �?       @      �?      @               @               @               @              �?              �?      �?              �?              �?      �?              �?              �?              �?               @              �?              �?              �?              �?      �?               @      �?              �?              �?      �?              �?      @               @      �?      @      @       @       @      �?      @      @      @       @      @      @      @      @      (@      @      (@      @      *@      *@       @      &@      ,@      2@      4@      2@      7@      =@      4@      @@     �A@      C@      ?@      C@      M@     �E@     �K@     �P@      S@     @U@     �Y@     @V@     @Y@     @Z@     ``@      a@     �b@     �c@     �e@     @k@      p@     �l@     pp@      r@     @v@     �w@     �v@     P{@     �}@     �@     X�@     ��@      �@     Ȉ@     H�@     ��@     \�@     H�@     X�@     ��@     ��@     ��@     \�@     ��@     �@     ֢@     �@     *�@     ��@     ��@     v�@     �@     ��@     ޴@     `�@     n�@     X�@     �@    ���@     �@    ���@    �*�@    �]�@    ���@    �A�@     �@    ���@    @��@    @c�@     ��@    @��@    @|�@    �}�@    @,�@    `
�@    ���@    ��@     6�@    @��@    `��@     ��@    Pj�@    �m�@    @y�@     ��@    @V�@     6�@    @| A    0A    ��A    �A     ;A    ��
A    �xA    d1A    ��A    �A    (�A    ��A    �<A    @�A    ��A    �!A    �a#A    V%A    ��'A    0�)A    ��,A    l�/A    �b1A    �'3A    �$5A    Vb7A    ��9A    w<A    t?A   �xPAA    ��BA   � )DA   �'EA   ���EA   ���FA    ��GA    ��IA   ���KA    F�NA   @�1PA   @ٹPA   �A�QA    K�QA   @�=PA    o�KA   �_"GA   �RCA   ��
CA    :�;A    c-2A    ��+A    x�%A    7!A    {A    4A    @�A    (A    �lA    �A    ;�@    �x�@    ���@    �v�@     �@      �?        
"
CNN2/weights/summaries/meanV��
&
CNN2/weights/summaries/stddev_1�K�=
!
CNN2/weights/summaries/maxf+�>
!
CNN2/weights/summaries/min����
�
 CNN2/weights/summaries/histogram*�	    іҿ   �lE�?      �@! �|�0B;�)�]��3~Z@2�_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:��u�w74���82���bȬ�0���VlQ.��7Kaa+��[^:��"��S�F !��T7����5�i}1�x?�x��>h�'��f�ʜ�7
������I��P=��pz�w�7��})�l a��ߊ4F����>M|Kվ��~]�[Ӿ��Zr[v�>O�ʗ��>f�ʜ�7
?>h�'�?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?�������:�               @      &@      6@     �O@     `b@     �h@      q@     �q@     pr@     �s@     �s@     �r@     0r@     @r@     �p@     0p@     q@     �n@     �l@     �h@     �g@     �f@     @b@     @a@     @b@     @]@      Z@     �[@     �Y@     �V@     @U@     @R@     �T@     �Q@      J@     �L@      D@     �F@     �A@     �A@      @@      <@      4@      6@      8@      8@      .@      2@      8@      .@      *@      0@      $@       @       @      0@      @      "@       @      �?      @              @      @      @               @       @      �?              �?       @              �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?              �?      �?              @       @              @              �?      @      @       @      �?      @      @      @      @      @       @      @      @      &@       @      "@       @       @      (@      *@      3@      4@      1@      6@      6@      9@      @@      ?@     �D@      ?@     �A@      E@     �K@     �K@      R@     @P@      P@      T@     �X@     �]@     �X@     �[@     �Z@      a@     �b@     �c@      f@     �h@      l@     �k@     �o@     �l@     �o@     �p@     0r@     �q@     0r@     `r@     @q@     �p@     �p@     �n@     `j@      b@      F@      ,@      �?        
!
CNN2/biases/summaries/mean�;�=
%
CNN2/biases/summaries/stddev_1AԳ<
 
CNN2/biases/summaries/maxL�>
 
CNN2/biases/summaries/minl�=
�
CNN2/biases/summaries/histogram*�	   �-=�?   ����?      @@!   �x�
@)��5�50�?2X����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:X              @      �?      @       @      @      @      @      @      �?        
�
CNN2/activations*�   @#�<@     D�A!&���n�A)DT�nǭA2�        �-���q=��z!�?�>��ӤP��>�
�%W�>R%�����>�u��gr�>��|�~�>���]���>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�*ǣ0?@�������:�           �!X�A              �?      �?              �?              @              �?       @              @      @      @      @       @      @      @       @       @      @      @      @      @      @      @      @       @      "@      .@      @      (@      @      1@      6@      2@      9@      1@      :@      9@      =@     �A@      ;@     �C@     �F@      M@     �P@      M@      R@     �Q@     @T@      V@      W@     �_@     `a@      c@      b@      e@     �d@     @f@     �j@      k@     �p@     @o@     �t@     pu@     �x@     �y@     {@     P}@     `�@     `�@     H�@     8�@      �@     ��@     0�@     ؐ@     8�@     ��@     `�@     ��@     Ě@     �@     �@     ��@     ��@     Z�@     ��@     p�@     *�@      �@     ��@     ��@     G�@     �@     �@     �@     y�@     ��@     ��@    ���@     ��@    ���@     h�@     �@    @��@     )�@    ���@     �@    �-�@     ��@    @@�@     $�@     ��@    ���@     ��@     }�@    `�@    @��@    �H�@    �u�@    �)�@     ��@    p�@    @|�@     ��@    ���@    X� A    ��A    @�A    �A    h�A    �`A    �A    @�A    �PA    +A    D6A    �jA    ��A    8�A    X] A    4�!A    t�#A    ��%A    N�'A    �*A    �,A    �./A    a1A    
�2A    ��4A    0 7A    �9A    ��<A    VK?A    v�@A   �֋AA    AA    ��@A    ��?A    �>A    �;>A    ��>A     {?A    }j>A    G�9A    ��2A    ��-A    ��&A    �Y!A    D�A    �A    ��	A    0A    @��@    `��@     ��@     ��@    ��@     ��@     9�@     !�@     Ҧ@     h�@     ��@     �{@     �h@      G@      �?        
�.
CNN2/batch_norm*�-	    |)�   ���6@     D�A!�.�Y�(�)"��˜A2�������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��ڿ�ɓ�i�:�AC)8g�E'�/��x>f^��`{>u��6
�>T�L<�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�������:�            K�7A    @��@   ���NA   �nŁA   ��yA    �50A    �-A    $�*A    �(A    �v&A    li$A    �"A    �� A    �A    ��A     >A    ��A    X�A    $�A    �A    �A     %A    ��	A    �/A    A    XA    �GA    @��@     y�@    ���@    @��@    Ё�@    ���@     ��@    @ �@    �J�@     ��@    �G�@    ���@    ���@    �U�@    ���@    @
�@    �^�@    ��@    @��@     ��@     ��@    @��@     ��@    �@�@    ��@      �@     c�@     K�@    �X�@     +�@     ,�@     ��@     �@     ��@     ��@     �@     c�@     ��@     �@     ��@     4�@     ^�@     ��@     �@     t�@     ��@     �@     ��@     ��@     x�@     ̑@     (�@     ��@     ��@     ��@     8�@     ��@     ȁ@     ��@     �{@     z@     w@      u@     s@     0r@      r@      n@      k@     �f@     @g@      e@     @b@     �^@     @]@     @\@      W@     @Z@      Q@     �Q@      S@      K@      L@     �D@      G@      F@      E@     �@@      C@      C@     �@@      4@      8@      ;@      .@      3@      0@      @      @      $@      $@      @      "@      @       @      @      @      @      @      @       @       @       @      @       @      �?      @       @              @              @       @      @      @      �?              �?               @               @      �?      �?              �?              �?              �?              �?      @              �?      �?              @               @      @      @      �?      �?      �?       @       @      @      �?       @      @      @      @      @      @      @      @      "@      @      @      @      @      $@      "@      .@      (@      *@      .@      .@      1@      :@      5@      2@      <@      =@      <@      <@      ?@      H@     �E@     �O@     �L@     �O@      P@     �T@     �R@     @V@     @Y@     �W@     �Z@     @\@     �`@     @_@     �e@     �h@     �f@      l@     @m@     �q@     �q@     @r@     `t@     0x@     �y@      ~@     �~@     p�@     ��@     ��@     Ї@     h�@     ��@     ��@     P�@     Ȓ@     ��@     ��@     8�@     �@     ؞@     t�@     v�@     |�@     <�@     �@     �@     �@     ��@     �@     x�@     p�@     ?�@     �@     ��@    ��@     ��@     (�@     �@     7�@     ��@     K�@    ���@    ��@    ���@    @p�@     ��@    @$�@    @h�@    �-�@    �v�@    �E�@    ���@     ��@     B�@    ���@    �z�@    �@    P��@    @��@    ���@     ��@    P�@    ���@    ���@    (zA    >A    �+A    �VA    p�	A    �;A    �#A    � A    $�A    ��A    �
A    �bA    �A    t�A    �!A    B�"A    ��$A    �&A    �(A    :�*A    #-A    6j/A    ��0A    t�1A    /N2A    ��2A    �3A    	?3A    g~3A    i�3A    �O4A    [�4A    �5A    &7A    ^�7A    37A    'k4A    �O0A    N�*A    >G&A    X�!A    |A    �!A    \wA    p�	A    �5A    ���@     e�@     ��@    ���@    ���@    ���@     ��@     Ͽ@     ��@     ,�@     $�@     ��@     ��@     �n@     @V@       @        
"
CNN3/weights/summaries/meanWt*�
&
CNN3/weights/summaries/stddev_1��=
!
CNN3/weights/summaries/max�>
!
CNN3/weights/summaries/mine6��
�
 CNN3/weights/summaries/histogram*�	   ���п   ����?      �@!��4�NU�)�נ�#q@2����ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��6�]���1��a˲���[��O�ʗ�����Zr[v��pz�w�7��})�l a��u`P+d�>0�6�/n�>8K�ߝ�>�h���`�>�ߊ4F��>1��a˲?6�]��?����?f�ʜ�7
?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�������:�              (@     �R@      h@     �x@     ؀@     ��@     �@     ��@     ��@     p�@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     @�@      �@     �~@      }@     Py@     �x@      t@     �t@     �r@     �o@     �m@     p@     �j@     �f@      b@      ^@     `b@      `@     @Y@      [@     �V@     �[@     @R@     �P@      K@      M@      K@      D@      @@     �A@      N@      =@      A@      9@      >@      5@      7@      7@      ,@      $@      &@      .@      @      (@      "@      "@      @       @      @      @      @      @       @      @      @               @      @      @       @       @      �?       @      �?      @       @      �?      �?              �?      @      �?              �?       @              �?              �?              �?              �?      �?              �?      �?      �?              �?              �?      @      �?       @               @       @      �?      @       @      @       @       @      @       @      @      @      @      @      &@       @      &@      @      $@      "@      5@      0@      *@      1@      2@      7@      2@      =@      ;@     �@@      C@     �A@     �F@     �E@      F@      J@     �F@     �Q@      T@      T@     �Q@     �W@      Y@      `@      ^@     `c@     �`@     �c@     `h@     `h@      j@     �n@     �p@     pq@     �t@     �u@     px@     �z@     �{@     �}@     @~@     ��@     Ȃ@     @�@     P�@     �@     ��@     �@     ��@     p�@     ��@     ��@     H�@     h�@     �{@     v@      d@      B@      3@       @        
!
CNN3/biases/summaries/mean�P�=
%
CNN3/biases/summaries/stddev_1���<
 
CNN3/biases/summaries/max�>
 
CNN3/biases/summaries/min݊�<
�
CNN3/biases/summaries/histogram*�	   �[�?   ``6�?      P@!  �~�@)z�o��k�?2x^�S���?�"�uԖ?}Y�4j�?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:x              �?       @               @      @      @      "@      *@      @      (@      @      �?      �?        
�
CNN3/activations*�   ��:@     D�A!��_ʓ�A)��4�k�A2�        �-���q=:�AC)8g>ڿ�ɓ�i>ہkVl�p>BvŐ�r>T�L<�>��z!�?�>��ӤP��>���m!#�>�4[_>��>
�}���>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@}w�˝M5@�i*`�n7@�6��9@o�=o�Z<@�������:�           ��߀A              �?              �?              �?      �?              �?       @              �?       @              �?              �?              �?       @               @               @              @      �?              �?      �?      @      �?       @      @      @      @      @      @      @      @      $@      @      &@      .@      "@      $@      @      ,@      (@      ,@      .@      4@      6@      3@      3@     �@@      8@      ?@      :@     �A@     �@@      F@      G@     �L@     �Q@     �H@     �Q@     �R@     @Y@     �U@     �U@     �^@     �b@      `@      c@      g@     �h@     �j@     �m@     �l@     �o@     �r@     �s@     �v@     pz@     �|@     p~@     ��@     ��@     X�@     P�@     ��@     �@     `�@     x�@     �@     ��@     ܔ@     �@     ��@     ��@     8�@     @�@     ��@     R�@     n�@     ڨ@     ֬@     z�@     ϰ@     �@     `�@     ɵ@     ��@     ~�@     ��@     n�@     ��@     ��@    �N�@     ��@     2�@     |�@     ��@     G�@    �;�@     @�@     ��@    ���@    �7�@     ��@    ���@    @��@    ���@    ���@     ��@     )�@    @�@    ���@     �@    ���@    ���@    ���@    @%�@    ���@     ��@    h�A    �'A     �A    xA    /	A    ��A    8AA    x�A    �A    �A    P�A    <�A    X�A    �A    �wA    �� A    &�!A    u#A    �%A    `�&A    �_(A    
�)A    �+A    ��,A    �-A    4�.A    �v/A    ,�.A    "8-A    ��*A    ��(A    �O&A    X�#A    $q!A    �A    ��A    8�A    �A    @bA    ��	A    دA    `s A    0��@    ��@    ��@     ��@    @��@     ��@    ��@     +�@     ��@     H�@     �s@      O@      .@      @        
�,
CNN3/batch_norm*�,	   �0?�   @��2@     D�A!8E�}�[AA)>Xb��A2��1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K��
�}�����4[_>������m!#���
�%W��T�L<��u��6
�������~�f^��`{�w`f���n�=�.^ol�K���7�>u��6
�>��ӤP��>�
�%W�>���m!#�>
�}���>X$�z�>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@u�rʭ�@�DK��@{2�.��@!��v�@زv�5f@��h:np@S���߮@)����&@a/5L��@v@�5m @��@�"@�ՠ�M�#@sQ��"�%@e��:�(@����t*@�}h�-@�x�a0@�����1@q��D�]3@�������:�           `nsA   `2�nA    `!!A    ��A    ��A    �$A    ܳA     }A    `�A    (�A    �=A    0YA    ��A    �nA    8:A    `IA     � A     3�@    �`�@    ��@    ���@    a�@    �_�@    ���@    ���@    ���@     8�@    ���@    @��@    ���@     $�@    ���@    @J�@     ��@     �@    @�@    �D�@    �~�@    ���@    ��@    ���@    ���@    �.�@     �@    ���@     Y�@     [�@     ��@     ��@     t�@     %�@     &�@     ��@     h�@     `�@     �@     ��@     4�@     ܡ@     ��@     f�@     ��@     (�@     H�@      �@     ��@     �@      �@     8�@     `�@     @�@     ��@     ��@     0�@      �@      ~@     p}@     Pw@     �u@     �s@     �q@      q@     @k@     �k@     �i@      h@      d@     @c@     �`@      `@     �W@     �X@     @W@     @W@      T@     �Q@     @S@     �P@     �H@     �H@     �C@      E@      @@     �D@     �@@      6@      8@      :@      =@      7@      2@      .@      (@      (@      "@      $@       @      @      "@      @      @      @      @      @      @      @       @      @       @      @      @       @      @               @       @               @              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?       @              �?      �?       @       @      �?       @              �?      @       @      @      �?       @              @      �?       @      @      @      "@      @      ,@      @      $@       @      *@       @      *@      "@      *@      0@      6@      3@      0@      0@      <@      ?@      A@     �D@      D@      C@      E@     �G@     �K@      K@     �J@     �N@     �S@     �T@      R@     @Y@     �^@      b@     `a@      a@     @b@     �g@      i@     `k@     0q@     @r@     pq@     �s@     pw@     �x@     �|@     @{@     (�@      �@     x�@      �@     ��@     �@     ��@     ��@     \�@     Ȓ@     0�@     �@     D�@     ��@     �@     ��@     ��@     ��@     V�@     ��@     ث@     P�@     2�@     ;�@     �@     E�@     �@     [�@     F�@    �-�@    ���@    �,�@    ���@    �p�@     ��@     ��@     ^�@    ���@     $�@    ���@    �3�@    ���@    ���@    �e�@     ��@    �x�@    ��@    `;�@    ���@    ���@    �v�@    `I�@    ���@    ���@    `o�@    �^�@    ���@    pS�@    ��@    pA    ��A    �tA    PmA    �oA    0�
A    h2A    ��A    8`A    ,�A    �yA    �?A    tA    �A    ��A    ��A    (�A    ~� A    ""A    <#A    P�#A    �P$A    x�$A    �#$A    N|#A    ��"A    \"A    �!A    �A     PA     :A    ��A    �`A    ��A    7A    HsA    `	A    0�A    xgA     y�@    ��@     ��@    ���@    @��@     �@     ��@    ��@     �@     <�@     �@     ��@     �_@      ;@      @        
 
accuracy_streaming_mean_1    ׆A