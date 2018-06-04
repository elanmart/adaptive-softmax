
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
local os      = require 'os'

local word2vec = require 'utils.word2vec'
local cmd      = torch.CmdLine('-', '-')

cmd:option('-nhid',    256,           'nhid')
cmd:option('-batch',   64,            'batch')
cmd:option('-niter',   1000,          'niter')
cmd:option('-cutoff',  '2000,10000',  'cutoff')
cmd:option('-ntoken',  100000,        'ntoken')

local config = cmd:parse(arg)

local cutoff = tablex.map(tonumber, stringx.split(config.cutoff, ','))
table.insert(cutoff, config.ntoken)

local decoder   = nn.AdaptiveSoftMax(config.nhid, cutoff, config.divval)
local criterion = nn.AdaptiveLoss(cutoff)

local input  = torch.FloatTensor(config.batch * config.niter, config.nhid)
local target = torch.LongTensor(config.batch * config.niter)

input:uniform(-0.1, 0.1)
target:random(1, config.ntoken)

decoder:cuda()
criterion:cuda()
input = input:cuda()
target = target:cuda()

idx = 1

timer = torch.Timer()

while idx < target:size()[1] do
    local x = input:narrow(1, idx, config.batch)
    local y = target:narrow(1, idx, config.batch)

    decoder:zeroGradParameters()

    decoder:setTarget(y)
    decoder:forward(x)
    criterion:forward(decoder.output, y)

    criterion:backward(decoder.output, y)
    decoder:backward(x, criterion.gradInput)

    idx = idx + config.batch
end

print(timer:time().real)