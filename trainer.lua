require 'dp'

paths.dofile('cnn_model.lua')

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training ImageNet (large-scale image classification) using an Alex Krizhevsky Convolution Neural Network')
cmd:text('Ref.: A. http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf')
cmd:text('B. https://github.com/facebook/fbcunn/blob/master/examples/imagenet/models/alexnet_cunn.lua')
cmd:text('Example:')
cmd:text('$> th alexnet.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--trainPath', '', 'Path to train set. Defaults to --dataPath/ILSVRC2012_img_train')
cmd:option('--validPath', '', 'Path to valid set. Defaults to --dataPath/ILSVRC2012_img_val')
cmd:option('--metaPath', '', 'Path to metadata. Defaults to --dataPath/metadata')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
 cmd:option('-weightDecay', 5e-4, 'weight decay')
cmd:option('--maxNormPeriod', 1, 'Applies MaxNorm Visitor every maxNormPeriod batches')
cmd:option('--momentum', 0.9, 'momentum') 
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')
cmd:option('--verbose', false, 'print verbose messages')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--LCN', false, 'use Local Constrast Normalization as in the original paper. Requires inn (imagine-nn)')
cmd:text()
opt = cmd:parse(arg or {})

table.print(opt)

if opt.LCN then
   assert(opt.cuda, "LCN only works with CUDA")
   require "inn"
end


--[[data]]--
local datasource = dp.ImageSource{load_size = opt.loadSize, sample_size = opt.sampleSize, train_path = opt.trainPath, valid_path = opt.validPath, meta_path = opt.metaPath, verbose = not opt.silent}

-- preprocessing function 
ppf = datasource:normalizePPF()

model = createModel(opt)

--[[Visitor]]--
local visitor = {}
-- the ordering here is important:
if opt.momentum > 0 then
   if opt.accUpdate then
      print"Warning : momentum is ignored with acc_update = true"
   end
   table.insert(visitor, 
      dp.Momentum{momentum_factor = opt.momentum}
   )
end
if opt.weightDecay and opt.weightDecay > 0 then
   if opt.accUpdate then
      print"Warning : weightdecay is ignored with acc_update = true"
   end
   table.insert(visitor, dp.WeightDecay{wd_factor=opt.weightDecay})
end
table.insert(visitor, 
   dp.Learn{
      learning_rate = opt.learningRate, 
      observer = dp.LearningRateSchedule{
         schedule={[1]=1e-2,[19]=5e-3,[30]=1e-3,[44]=5e-4,[53]=1e-4}
      }
   }
)
if opt.maxOutNorm > 0 then
   table.insert(visitor, dp.MaxNorm{
      max_out_norm = opt.maxOutNorm, period=opt.maxNormPeriod
   })
end

--[[Propagators]]--
train = dp.Optimizer{
   loss = dp.NLL(),
   visitor = visitor,
   feedback = dp.Confusion(),
   sampler = dp.RandomSampler{
      batch_size=opt.batchSize, epoch_size=opt.trainEpochSize, ppf=ppf
   },
   progress = opt.progress
}
valid = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.Confusion(),  
   sampler = dp.Sampler{
      batch_size=math.round(opt.batchSize/10),
      ppf=ppf
   }
}

--[[Experiment]]--
xp = dp.Experiment{
   model = model,
   optimizer = train,
   validator = valid,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.maxTries
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

print"nn.Modules :"
print(model:toModule(datasource:trainSet():sub(1,32)))

xp:run(datasource)
