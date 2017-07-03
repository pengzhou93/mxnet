# Train #

# Evaluate #

## vgg16_300x300 ##
Precision
---------
mAP: 0.775602748312	
Details
-------
aeroplane: 0.794403722721
bicycle: 0.841639064523
bird: 0.754011578683
boat: 0.6973124303
bottle: 0.501906619473
bus: 0.863830451129
car: 0.858282875308
cat: 0.887905341782
chair: 0.590021874317
cow: 0.787640326534
diningtable: 0.783868056827
dog: 0.854511368732
horse: 0.886023033249
motorbike: 0.852128201334
person: 0.794141117231
pottedplant: 0.524522517948
sheep: 0.778348857712
sofa: 0.804841668453
train: 0.876750046204
tvmonitor: 0.779965813785

Model
-----

```python
    parser.add_argument('--network', dest='network', type=str,
                        default='legacy_vgg16_ssd',
                        help='which network to use')
    parser.add_argument('--prefix', dest='prefix', help='load model prefix',
                        default=os.path.join(os.getcwd(),
                        './pretrain_models/mxnet/vgg16_reduced_300x300/vgg16_ssd_300_voc0712_trainval', 'ssd_'),
                        type=str)

``` 

## resnet-50_512x512 ##
Precision
---------
mAP: 0.788749797011
Details
-------
aeroplane: 0.815747148925
bicycle: 0.868776531766
bird: 0.810832302478
boat: 0.71446810869
bottle: 0.584697560606
bus: 0.874317621608
car: 0.878852863445
cat: 0.889885167082
chair: 0.614850942819
cow: 0.848481604782
diningtable: 0.677985922983
dog: 0.868315001138
horse: 0.876202658287
motorbike: 0.871590152471
person: 0.797682439339
pottedplant: 0.511837258564
sheep: 0.807124477437
sofa: 0.790995673301
train: 0.870079879571
tvmonitor: 0.802272624936

Model
-----

```bash
    parser.add_argument('--network', dest='network', type=str,
                        default='resnet50',
                        help='which network to use')
    parser.add_argument('--prefix', dest='prefix', help='load model prefix',
                        default=os.path.join(os.getcwd(),
                        './pretrain_models/mxnet/resnet50_512x512', 'ssd_'),
                        type=str)
``` 


# Md mxnet source code #
1. mxnet/module/module.py:578
```python
elif hasattr(data_batch, "label") and data_batch.label and self._label_shapes:
# elif hasattr(data_batch, "label") and data_batch.label:
``` 




