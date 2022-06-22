import torch

print( torch.cuda.is_available() )
#>> True

print( torch.cuda.device_count() )
#>> 1

print( torch.cuda.current_device() )
#>> 0

print( torch.cuda.device(0) )
#>> <torch.cuda.device at 0x7efce0b03be0>

print( torch.cuda.get_device_name(0) )
#>> 'GeForce GTX 950M'