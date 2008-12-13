/*
 * Encog Neural Network and Bot Library for Java v1.x
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008, Heaton Research Inc., and individual contributors.
 * See the copyright.txt in the distribution for a full listing of 
 * individual contributors.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 */
package org.encog.nlp.memory;

/**
 *
 * @author  jheaton
 */
public class ConceptNotFoundException extends java.lang.Exception {
    
    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;


	/**
     * Creates a new instance of <code>ConceptNotFound</code> without detail message.
     */
    public ConceptNotFoundException() {
    }
    
    
    /**
     * Constructs an instance of <code>ConceptNotFound</code> with the specified detail message.
     * @param msg the detail message.
     */
    public ConceptNotFoundException(String msg) {
        super(msg);
    }
}
