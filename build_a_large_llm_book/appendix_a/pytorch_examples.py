import torch

# Tensor = mathematical concept that generalizes vectors and matrices to 
# potentially higher dimensions 

tensor0d = torch.tensor(1)
print (tensor0d)


tensor1d = torch.tensor([1,2,3])
print (tensor1d)
print (tensor1d.type)

tensor2d = torch.tensor([[1,2],[3,4]])
print (tensor2d)

tensor3d = torch.tensor([[[1,2], [3,4]], [[5,6], [7,8]]])
print (tensor3d)

floatvec = torch.tensor([1.0, 2.0, 3.0])
print (floatvec.dtype)

floatvec2 = tensor1d.to(torch.float32)
print (floatvec2.dtype)

tensor2d = torch.tensor([[1,2,3],[4,5,6]])
print ("New 2D tensor is:")
print (tensor2d)
print ("--------")

print ("Shape = rows and columns")
print ("Shape = " + str(tensor2d.shape))
print ("--------")

print ("Reshape = switch columns and rows")
print (tensor2d.reshape(3,2))
print ("---------")

print ("Using view command to reshape tensor")
print (tensor2d.view(3,2))
print ("---------")

print ("Tensor transposition using T")
print (tensor2d.T)
print ("---------")

print ("Matrix multiplication using matmul")
print (tensor2d.matmul(tensor2d.T))
print ("---------")

print ("Matrix multiplication using @")
print (tensor2d @ tensor2d.T)







