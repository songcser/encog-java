/*
 * Encog(tm) Core v3.0 Unit Test - Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 
 * Copyright 2008-2011 Heaton Research, Inc.
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
package org.encog.persist;

import java.io.File;
import java.io.IOException;

import junit.framework.Assert;
import junit.framework.TestCase;

import org.encog.ml.svm.KernelType;
import org.encog.ml.svm.SVM;
import org.encog.ml.svm.SVMType;
import org.encog.ml.svm.training.SVMTrain;
import org.encog.neural.art.ART1;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.XOR;
import org.encog.util.obj.SerializeObject;

public class TestPersistSVM extends TestCase {
	
	public final String EG_FILENAME = "encogtest.eg";
	public final String EG_RESOURCE = "test";
	public final String SERIAL_FILENAME = "encogtest.ser";
	
	private SVM create()
	{
		NeuralDataSet training = new BasicNeuralDataSet(XOR.XOR_INPUT,XOR.XOR_IDEAL);
		SVM result = new SVM(2,SVMType.EpsilonSupportVectorRegression,KernelType.RadialBasisFunction);
		final SVMTrain train = new SVMTrain(result, training);
		train.iteration();
		return result;
	}
	
	public void testPersistEG()
	{
		SVM network = create();

		EncogDirectoryPersistence.saveObject(new File(EG_FILENAME), network);
		SVM network2 = (SVM)EncogDirectoryPersistence.loadObject(new File(EG_FILENAME));
		validate(network2);
	}
	
	public void testPersistSerial() throws IOException, ClassNotFoundException
	{
		SVM network = create();
		
		SerializeObject.save(SERIAL_FILENAME, network);
		SVM network2 = (SVM)SerializeObject.load(SERIAL_FILENAME);
				
		validate(network2);
	}
	
	public void testPersistSerialEG() throws IOException, ClassNotFoundException, InstantiationException, IllegalAccessException
	{
		SVM network = create();
		
		SerializeObject.save(SERIAL_FILENAME, network);
		SVM network2 = (SVM)SerializeObject.load(SERIAL_FILENAME);
				
		validate(network2);
	}
	
	private void validate(SVM svm)
	{
		Assert.assertEquals(KernelType.RadialBasisFunction, svm.getKernelType());
		Assert.assertEquals(SVMType.EpsilonSupportVectorRegression, svm.getSVMType());
		Assert.assertEquals(4, svm.getModel().SV.length);
	}
}
