import java.util.Scanner;
public class NeuralNetwork{ 
    
    static Layer[] layers;
    static TrainingData[] tDataSet; 

    public static void main(String[] args) {
        Neuron.setRangeWeight(-1,1); 
        layers = new Layer[3];
    	layers[0] = null; // Input Layer 0,6
    	layers[1] = new Layer(6,6); // Hidden Layer 2,6
    	layers[2] = new Layer(6,2); // Output Layer 6,1    

        //Create the training Data
        CreateTrainingData();
    	
        System.out.println("============");
        System.out.println("Expected output");
        System.out.println("============");
        for(int i = 0; i < tDataSet.length; i++) {
           // System.out.println(layers[2].neurons[0].value);
			System.out.print(tDataSet[i].expectedOutput[0]);
            System.out.print(" "); 
            System.out.print(tDataSet[i].expectedOutput[1]);
            System.out.println(); 

		} 

        train(1000000, 0.05f);

        System.out.println("============");
        System.out.println("Output after training");
        System.out.println("============");
        for(int i = 0; i < tDataSet.length; i++) {
            forward(tDataSet[i].data);
            System.out.print(layers[2].neurons[0].value); 
            System.out.print(" ");
            System.out.print(layers[2].neurons[1].value);
            System.out.println();
        }
    }
    public static void CreateTrainingData() {
        Scanner myObj = new Scanner(System.in);
        float[][] input= new float[2][6];  
        float[][] output= new float[2][2];
        for(int i=0;i<2;i++) 
        {   
            for(int j=0;j<6;j++) 
            {   
                input[i][j]=(float)myObj.nextDouble();
            }
        } 
        // for(int i=0;i<2;i++) 
        // {   
        //     for(int j=0;j<6;j++) 
        //     {   
        //         System.out.print(input[i][j]); 
        //         System.out.print(" ");
        //     } 
        //     System.out.println();
        // } 
        // float[] input1=new float[]{(float)36.5,1,0,0,1,0};
        // float[] input2=new float[]{(float)36.0,0,0,1,0,1};
        // float[] input3=new float[]{(float)37.0,0,1,0,1,0};
        // float[] input4=new float[]{(float)37.5,1,1,0,1,0}; 

        // float[] expectedOutput1=new float[]{1,0}; 
        // float[] expectedOutput2=new float[]{0,0}; 
        // float[] expectedOutput3=new float[]{1,1}; 
        // float[] expectedOutput4=new float[]{0,1};
        
        // tDataSet = new TrainingData[4];
        // tDataSet[0] = new TrainingData(input1, expectedOutput1);
        // tDataSet[1] = new TrainingData(input2, expectedOutput2);
        // tDataSet[2] = new TrainingData(input3, expectedOutput3);
        // tDataSet[3] = new TrainingData(input4, expectedOutput4);        

        for(int i=0;i<2;i++) 
        {   
            for(int j=0;j<2;j++) 
            {   
                output[i][j]=(float)myObj.nextDouble();
            }
        }
 
        tDataSet = new TrainingData[2];
        for(int i=0;i<2;i++)
        {
            tDataSet[i] = new TrainingData(input[i],output[i]);
        }        
    }

    public static void forward(float[] inputs) {
    	// First bring the inputs into the input layer layers[0]
    	layers[0] = new Layer(inputs);
    	
        for(int i = 1; i < layers.length; i++) {

        	for(int j = 0; j < layers[i].neurons.length; j++) {
        		float sum = 0;
        		for(int k = 0; k < layers[i-1].neurons.length; k++) {
        			sum += layers[i-1].neurons[k].value*layers[i].neurons[j].weights[k];
        		}
        		//sum += layers[i].neurons[j].bias; // TODO add in the bias 
        		layers[i].neurons[j].value = StatUtil.Sigmoid(sum);
        	}
        } 	
    }
    
    public static void backward(float learning_rate,TrainingData tData) {
    	
    	int number_layers = layers.length;
    	int out_index = number_layers-1;
    	
    	// Update the output layers 
    	// For each output

		//-----------------Mathematical part -> To be ignored ----------------------------------------------------------------
    	for(int i = 0; i < layers[out_index].neurons.length; i++) {
    		// and for each of their weights
    		float output = layers[out_index].neurons[i].value;
    		float target = tData.expectedOutput[i];
    		float derivative = output-target;
    		float delta = derivative*(output*(1-output));
    		layers[out_index].neurons[i].gradient = delta;
    		for(int j = 0; j < layers[out_index].neurons[i].weights.length;j++) { 
    			float previous_output = layers[out_index-1].neurons[j].value;
    			float error = delta*previous_output;
    			layers[out_index].neurons[i].cache_weights[j] = layers[out_index].neurons[i].weights[j] - learning_rate*error;
    		}
    	}
    	
    	//Update all the subsequent hidden layers
    	for(int i = out_index-1; i > 0; i--) {
    		// For all neurons in that layers
    		for(int j = 0; j < layers[i].neurons.length; j++) {
    			float output = layers[i].neurons[j].value;
    			float gradient_sum = sumGradient(j,i+1);
    			float delta = (gradient_sum)*(output*(1-output));
    			layers[i].neurons[j].gradient = delta;
    			// And for all their weights
    			for(int k = 0; k < layers[i].neurons[j].weights.length; k++) {
    				float previous_output = layers[i-1].neurons[k].value;
    				float error = delta*previous_output;
    				layers[i].neurons[j].cache_weights[k] = layers[i].neurons[j].weights[k] - learning_rate*error;
    			}
    		}
    	}
    	
    	// Here we do another pass where we update all the weights
    	for(int i = 0; i< layers.length;i++) {
    		for(int j = 0; j < layers[i].neurons.length;j++) {
    			layers[i].neurons[j].update_weight();
    		}
    	}
    	
    }
    
    // This function sums up all the gradient connecting a given neuron in a given layer
    public static float sumGradient(int n_index,int l_index) {
    	float gradient_sum = 0;
    	Layer current_layer = layers[l_index];
    	for(int i = 0; i < current_layer.neurons.length; i++) {
    		Neuron current_neuron = current_layer.neurons[i];
    		gradient_sum += current_neuron.weights[n_index]*current_neuron.gradient;
    	}
    	return gradient_sum;
    }
    public static void train(int training_iterations,float learning_rate) {
    	for(int i = 0; i < training_iterations; i++) {
    		for(int j = 0; j < tDataSet.length; j++) {
    			forward(tDataSet[j].data);
    			backward(learning_rate,tDataSet[j]);
    		}
    	}
    }

}


//output-> 1-0.5999=>"no" , 1-0.899999=>"yes"
//
