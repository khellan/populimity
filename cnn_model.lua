-- This is the AlexNet model from dp
function buildModel(opt)
   local features = nn.Concat(2)
   local fb1 = nn.Sequential() -- branch 1
   fb1:add(nn.SpatialConvolutionMM(3,48,11,11,4,4,2,2))       -- 224 -> 55
   fb1:add(nn.ReLU())
   if opt.LCN then
      fb1:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 2))
   end
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   
   fb1:add(nn.SpatialConvolutionMM(48,128,5,5,1,1,2,2))       --  27 -> 27
   fb1:add(nn.ReLU())
   if opt.LCN then
      fb1:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 2))
   end
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   
   fb1:add(nn.SpatialConvolutionMM(128,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   
   fb1:add(nn.SpatialConvolutionMM(192,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   
   fb1:add(nn.SpatialConvolutionMM(192,128,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
   fb1:add(nn.Copy(nil, nil, true)) -- prevents a newContiguous in SpatialMaxPooling:backward()

   local fb2 = fb1:clone() -- branch 2
   for k,v in ipairs(fb2:findModules('nn.SpatialConvolutionMM')) do
      v:reset() -- reset branch 2's weights
   end

   features:add(fb1)
   features:add(fb2)

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   
   classifier:add(nn.Linear(4096, 2)) -- Second parameter is the output layer dimension
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)
   -- wrap the nn.Module in a dp.Module

   return model
end

function createModel(opt)
   model = dp.Module{
      module = buildModel(opt),
      input_view = 'bchw', 
      output_view = 'bf',
      output = dp.ClassView()
   }
   return model
end
