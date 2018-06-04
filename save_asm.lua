
require 'math'
require 'cutorch'
require 'nn'
require 'cunn'
require 'rnnlib'

local tablex  = require 'pl.tablex'
local stringx = require 'pl.stringx'
local tnt     = require 'torchnet'
local optim   = require 'optim'
local data    = require 'data'
local utils   = require 'utils'
local json    = require 'json'

local word2vec = require 'utils.word2vec'
local cmd      = torch.CmdLine('-', '-')

cmd:option('-nhid',    32,                        'Number of hidden variables per layer')
cmd:option('-batch',   4096,                      'batchsize')
cmd:option('-cutoff',  '10,50,100',               'Cutoff for AdaptiveSoftMax')
cmd:option('-path',    '/tmp/AdaptiveSoftMax.t7', 'path to save the model')

local config = cmd:parse(arg)
local cutoff = tablex.map(tonumber, stringx.split(config.cutoff, ','))

local decoder   = nn.AdaptiveSoftMax(config.nhid, cutoff, config.divval)
local criterion = nn.AdaptiveLoss(cutoff)
local input     = torch.Tensor(config.batch, config.nhid)
local target    = torch.LongTensor(config.batch)

input:uniform(-0.1, 0.1)
target:random(1, 100)

decoder:zeroGradParameters()
decoder:setTarget(target)
decoder:forward(input)
criterion:forward(decoder.output, target)

criterion:backward(decoder.output, target)
decoder:backward(input, criterion.gradInput)

local logprob = decoder:getLogProb(input)

torch.save('/tmp/AdaptiveSoftMax.t7', {decoder   = decoder, 
                                       criterion = criterion, 
                                       input     = input, 
                                       target    = target,
                                       logprob   = logprob})
