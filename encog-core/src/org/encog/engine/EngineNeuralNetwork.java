/*
 * Encog(tm) Core v2.5 - Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 
 * Copyright 2008-2010 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */

package org.encog.engine;

/**
 * An interface that defines a neural network. Mainly adds the ability to
 * encode/decode weights to/from a double array.
 * 
 */
public interface EngineNeuralNetwork extends EngineMachineLearning {
	
	/**
	 * Decode an array to the neural network weights.
	 * @param data The data to decode.
	 */
	void decodeNetwork(double[] data);

	/**
	 * Encode the neural network weights to an array.
	 * @return The encoded neural network.
	 */
	double[] encodeNetwork();

	/**
	 * @return The length of the encoded array.
	 */
	int getEncodeLength();
}
