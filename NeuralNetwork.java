import java.io.*;
import java.util.StringTokenizer;

public class NeuralNetwork{ 
    
    static Layer[] layers;
    static TrainingData[] tDataSet; 

    public static void main(String[] args) throws FileNotFoundException,IOException {
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
    public static void CreateTrainingData() throws IOException {
        float[][] input= new float[120][6];  
        float[][] output= new float[120][2];
       //---------------File handling----------------------------
       FileReader file=new FileReader("C:\\Users\\snigd\\Desktop\\docs\\NeuralNetwork\\input.txt"); 
       BufferedReader bf= new BufferedReader(file); 
       
       String st=bf.readLine(); 
       int i=0; 
       while(st!=null){   
           
           StringTokenizer stn= new StringTokenizer(st);
           Double i0=Double.parseDouble(stn.nextToken()); 
           String str2=stn.nextToken();
           int i1; 
           if(str2.equals("yes")) 
           { 
             i1=1;
           } 
           else 
           {    
             i1=0;
           }

           String str3=stn.nextToken();
           int i2; 
           if(str3.equals("yes")) 
           {
             i2=1;
           } 
           else 
           {    
             i2=0;
           }
           String str4=stn.nextToken(); 
           Double i3=0.00; 
           if(str4.equals("yes")) 
           {
             i3+=1.00;
           }
           String str5=stn.nextToken();
           Double i4=0.00; 
           if(str5.equals("yes")) 
           {
             i4+=1.00;
           }
           String str6=stn.nextToken(); 
           Double i5=0.00; 
           if(str6.equals("yes")) 
           {
             i5+=1.00; 
           }
           String str7=stn.nextToken(); 
           Double i6; 
           if(str7.equals("yes")) 
           { 
             i6=1.00; 
           } 
           else 
           {    
             i6=0.00;
           }
           String str8=stn.nextToken(); 
           Double i7; 
           if(str8.equals("yes")) 
           {
             i7=(Double)1.00;
           }
           else 
           {    
             i7=0.00;
           } 
           

        //    System.out.println("Reading from file");
        // System.out.println(i0+" "+str2+" "+str3+" "+str4+" "+str5+" "+str6);
        // System.out.println(i0+" "+i1+" "+i2+" "+i3+" "+i4+" "+i5); 
        input[i][0]=i0.floatValue(); 
        input[i][1]=i1; 
        input[i][2]=i2; 
        input[i][3]=i3.floatValue(); 
        input[i][4]=i4.floatValue(); 
        input[i][5]=i5.floatValue(); 
        
        output[i][0]=i6.floatValue(); 
        output[i][1]=i7.floatValue();

        st=bf.readLine();
        i++;
       }	
       
    //    //For Input
    //     for(int i=0;i<2;i++) 
    //     {   
    //         for(int j=0;j<6;j++) 
    //         {   
    //             input[i][j]=(float)myObj.nextDouble();
    //         }
    //     } 

        
    //     //Expected Output
    //     for(int i=0;i<2;i++) 
    //     {   
    //         for(int j=0;j<2;j++) 
    //         {   
    //             output[i][j]=(float)myObj.nextDouble();
    //         }
    //     }
    //     //-----------------------------------------------------------
        
    
    // for( i=0;i<120;i++) 
    // {   
    //     System.out.println(input[i][0]+" "+input[i][1]+" "+input[i][2]+" "+input[i][3]+" "+input[i][4]+" "+input[i][5]); 
    // }
        tDataSet = new TrainingData[120];
        i=0;
        for(;i<120;i++)
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
