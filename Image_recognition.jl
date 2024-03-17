#Michał Wera 189001 WNO project
using Flux, Images, FileIO,Plots
using Flux: crossentropy, onecold, onehotbatch, train!
using LinearAlgebra, Random, Statistics
using MLDatasets: CIFAR10
using BSON: @save,@load
using ImageView

# Convert image to array that can be correctly read by model
function imgtoarray(image,arr1)

    for k in 1:3
        for j in 1:32
            for i in 1:32
            arr1[j,i,k,1] = image[k,i,j]
            end
        end
    end
    return arr1
end



label_names = 
 [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


#Loading data set(for training and testing) that contains 10 types of "Images" with correct labels
train_set = CIFAR10(Tx=Float32, split=:train)
test_set = CIFAR10(Tx=Float32, split=:test)

#size(train_set.targets)
#size(train_set.features)

train_images = Array{Float32,4}(undef,32,32,3,50000)
train_images = train_set.features
train_labels = Array{Float32,1}(undef,50000)
train_labels = train_set.targets

test_images = Array{Float32,4}(undef,32,32,3,10000)
test_images = test_set.features
test_labels = Array{Float32,1}(undef,10000)
test_labels = test_set.targets


train_labels = Flux.onehotbatch(train_labels,0:9)

train_loader = Flux.Data.DataLoader((train_images,train_labels), batchsize=128, shuffle=true)


#################################################################################################
###############......AI MODEL..............######################################################
################################################################################################
ai_model = Chain(
    #Layer 1 - 2xConv Layer
    Conv((3,3), 3=>32 ,pad = SamePad() ,relu),          
    BatchNorm(32, relu),
    Conv((3,3), 32=>32 ,pad = SamePad()  ,relu),
    BatchNorm(32, relu),
    MaxPool((2, 2)),              #Batchnorm and Dropout to prevent overfitting the model
    Dropout(0.2),
    #Layer 2 - 2xConv Layer
    Conv((3,3),32=>64,pad = SamePad(), relu),
    BatchNorm(64, relu),
    Conv((3,3),64=>64,pad = SamePad(), relu),
    BatchNorm(64, relu),
    MaxPool((2, 2)), 
    Dropout(0.3),
    #Layer 3 - 2xConv Layer
    Conv((3,3),64=>128,pad = SamePad(), relu),
    BatchNorm(128, relu),
    Conv((3,3),128=>128,pad = SamePad(), relu),
    BatchNorm(128, relu), 
    MaxPool((2,2)),                         
    Dropout(0.4),
    ######
    Flux.flatten,               #flattening the data 
    Dropout(0.5),
    #Layer 4 - 2xDense Layer
    Dense(2048,128),            #passing values through fully connected layers
    BatchNorm(128, relu),                         
    #######
    Dense(128,10),                        
    #######
    softmax                     #by using softmax final 10 outputs will sum up to 1(100%)
)
###################################################################################################
#quick model check before learning #######################################################################################
xtestt = rand(Float32,32,32,3,1)

ai_model(xtestt)|>size    #should be 10*X output



############Loading model
@load "alfa1.bson" ai_model
##################################################################################################
##################################################################################################
##################################################################################################
#TESTING SET ACCURACY:

function test_accuracy(testimages,testlabels,ai_model)
    y_hat_raw = ai_model(testimages)
    y_hat = onecold(y_hat_raw)
    y = (testlabels).+1
    testacc = mean(y_hat.==y)
    print("Accuracy on test_set: ")
    print(testacc, "\n") 
    return testacc
end


#test_accuracy(test_images,test_labels,ai_model)
## best reached accuracy on test_set = 0.8612



###################################################################################
#.............TRAINING PROCESS.....################################################
###################################################################################

function trening(ai_model,train_loader,test_images,test_labels,train_images,train_labels)
    #parameters
    opt = ADAM(0.001)  #change learning step depending on results
    accuracy(x, y) = mean(onecold(ai_model(x)) .== onecold(y))
    loss(x,y) = crossentropy(ai_model(x),y)
    global bestacc = 0.8612

    ##########training #######################################################
    @info("BEGGINING TRAINING PROCESS:")
    epochs = 1 # number of times whole data set will be passed through model             
    @time for epoch in 1:epochs
        Flux.train!(loss,Flux.params(ai_model),train_loader,opt)
        acc = test_accuracy(test_images,test_labels,ai_model)
        train_loss = loss(train_images,train_labels)
        println("Epoch = $epoch : Training Loss =$train_loss Acc= $acc")
        if acc>bestacc
            @save "betaacc.bson" ai_model
        end 
    end
end


#trening(ai_model,train_loader,test_images,test_labels,train_images,train_labels)








###################################################################################
###################################################################################
###################################################################################
###################################################################################
#insight into individual images from set with prediction 


#y_hat_raw = ai_model(test_images)
#imgidx = 2162
#zpliku = CIFAR10.convert2image(test_set[imgidx][1])
#tempik = onecold(y_hat_raw[:,imgidx])
#print("prediction: ")
#print(label_names[tempik])
#print("\n label: ")
#print(label_names[test_set[imgidx][2]+1])



####################################################################################


####testing outside images
function customtest(name,ai_model,label_names)
    base_path = pwd()
    imgpath = base_path*"\\PetImages\\"*name*".jpg"
    img = load(imgpath)
    img_square = imresize(img, (32, 32))
    imgtof = channelview(img_square)
    input_arr = Array{Float32,4}(undef,32,32,3,1)
    final_in = imgtoarray(imgtof,input_arr)
    pred_num = ai_model(final_in)
    prediction = label_names[onecold(pred_num)]
    imshow(img_square,canvassize = (350,350),name = "Prediction:  "*prediction[1])
    print("Prediction :")
    print(prediction,"\n")
end

#customtest("h1",ai_model,label_names)
#########################################################




