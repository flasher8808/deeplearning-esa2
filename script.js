//console.log('Hello TensorFlow');

// Zufallszahlen erzeugen
function generateRandomNumbers(count, min, max) {
  const randomNumbers = [];
  for (let i = 0; i < count; i++) {
    // Math.random() erzeugt einen Wert zwischen 0 (einschließlich) und 1 (ausschließlich)
    const randomValue = Math.random() * (max - min) + min;
    randomNumbers.push(randomValue);
  }
  return randomNumbers;
}

const count = 100;
const min = -2;
const max = 2;

const xValues = generateRandomNumbers(count, min, max);
let yValues = [];
for (let i = 0; i < count; i++) {
  yValues.push(calcYValue(xValues[i]));
}
//console.log("x-Werte:");
//console.log(xValues);
//console.log("y-Werte:");
//console.log(yValues);

// Y-Werte berechnen
function calcYValue (xval) {
  let yval = 0.5*(xval+0.8)*(xval+1.8)*(xval-0.2)*(xval-0.3)*(xval-1.9)+1;
  // Funktion y(x) = 0.5*(x+0.8)*(x+1.8)*(x-0.2)*(x-0.3)*(x-1.9)+1
  //console.log(yval);
  return yval;
}


function combineArrays(arr1, arr2) {
    // Sicherstellen, dass beide Arrays die gleiche Länge haben
    if (arr1.length !== arr2.length) {
        throw new Error('Die Arrays müssen die gleiche Länge haben.');
    }
    
    // Verwende map, um ein Array von Wertepaaren zu erstellen
    return arr1.map((value, index) => [value, arr2[index]]);
}

// ARRAYS ZUSAMMENFUEHREN

//let combined = combineArrays(xValues, yValues);
//console.log("Kombiniertes Array:");
//console.log(combined);


function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1)); // Wähle einen zufälligen Index aus 0 bis i
    // Tausche die Elemente an den Positionen i und j
    const temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
}

//console.log("x-Werte original:");
//console.log(xValues);


// Array mischen
shuffleArray(xValues);

console.log("x-Werte gemischt:");
console.log(xValues);

let testdaten = xValues.slice(0, count/2);
let trainingsdaten = xValues.slice(count/2);

console.log("Testdaten:");
console.log(testdaten);
console.log("Trainingsdaten:");
console.log(trainingsdaten);

let testdatenY = [];
let trainingsdatenY = [];

for (let i = 0; i < testdaten.length; i++) {
  testdatenY.push(calcYValue(testdaten[i]));
}

for (let i = 0; i < trainingsdaten.length; i++) {
  trainingsdatenY.push(calcYValue(trainingsdaten[i]));
}



console.log("TEST RAUSCHEN");
function generateNormalRandom(mu = 0, sigma = 1) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0 * sigma + mu;
}

function addGaussianNoise(data, variance) {
    const standardDeviation = Math.sqrt(variance);
    return data.map(value => value + generateNormalRandom(0, standardDeviation));
}


// Varianz des Rauschens
const variance = 0.05;

// Füge dem Array Gaußsches Rauschen hinzu
let trainingsdatenY_rausch = addGaussianNoise(trainingsdatenY, variance);
let testdatenY_rausch = addGaussianNoise(testdatenY, variance);


console.log("Trainingsdaten verrauscht");
console.log(trainingsdatenY_rausch);



console.log("Testdaten Y ");
console.log(testdatenY);
console.log("Testdaten Y verrauscht");
console.log(testdatenY_rausch);

console.log("Trainingsdaten");
console.log(trainingsdaten);
console.log("Trainingsdaten Y");
console.log(trainingsdatenY);


// JSON Objekte erzeugen
let tempTrainingsdaten = trainingsdaten.map((value, index) => ({ x: value, y: trainingsdatenY[index]}));
let jsonTrainingsdaten = tempTrainingsdaten;

console.log(jsonTrainingsdaten);


let jsonTestdaten;
let jsonTrainingsdatenVerrauscht;
let jsonTestdatenVerrauscht;







// Visualisierung Diagramm oben
/*
const ctx = document.getElementById('chart_values').getContext('2d');
const myChart = new Chart(ctx, {
  type: 'scatter',
  data: {
    datasets: [{
      label: 'Random Values',
      data: xValues.map((value, index) => ({ x: value, y: calcYValue(value) })),
      backgroundColor: 'rgba(75, 192, 192, 0.8)',
      pointRadius: 3
    }]
  },
  options: {
    plugins: {
      title: {
          display: true,
          text: 'Zufällige x-Wert mit berechneten y-Werten'
      }
    },
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        title: {
          display: true,
          text: 'x-Achse'
        }
      },
      y: {
        beginAtZero: true,
        min: -2,
        max: 3,
        title: {
          display: true,
          text: 'y-Achse'
        }
      }
    }
  }
});
*/

// Visualisierung Diagramm ohne Rauschen links
const ctx_ohneRauschen = document.getElementById('r1-ohneRauschen').getContext('2d');
const ChartohneRauschen = new Chart(ctx_ohneRauschen, {
  type: 'scatter',
  data: {
    datasets: [{
      label: 'Trainingsdaten',
      data: trainingsdaten.map((value, index) => ({ x: value, y: calcYValue(value) })),
      backgroundColor: 'rgba(192, 75, 75, 0.8)',
      pointRadius: 3
    },
    {
      label: 'Testdaten',
      data: testdaten.map((value, index) => ({ x: value, y: calcYValue(value) })),
      backgroundColor: 'rgba(75, 192, 192, 0.8)',
      pointRadius: 3
    }]
  },
  options: {
    plugins: {
      title: {
          display: true,
          text: 'Trainings- und Testdaten ohne Rauschen'
      }
    },
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        title: {
          display: true,
          text: 'x-Achse'
        }
      },
      y: {
        beginAtZero: true,
        min: -2,
        max: 3,
        title: {
          display: true,
          text: 'y-Funktionswerte'
        }
      }
    }
  }
});


// Visualisierung Diagramm mit Rauschen rechts
const ctx_mitRauschen = document.getElementById('r1-mitRauschen').getContext('2d');
const ChartmitRauschen = new Chart(ctx_mitRauschen, {
  type: 'scatter',
  data: {
    datasets: [{
      label: 'Trainingsdaten',
      data: trainingsdaten.map((value, index) => ({ x: value, y: trainingsdatenY_rausch[index] })),
      backgroundColor: 'rgba(192, 75, 75, 0.8)',
      pointRadius: 3
    },
    {
      label: 'Testdaten',
      data: testdaten.map((value, index) => ({ x: value, y: testdatenY_rausch[index] })),
      backgroundColor: 'rgba(75, 192, 192, 0.8)',
      pointRadius: 3
    }]
  },
  options: {
    plugins: {
      title: {
          display: true,
          text: 'Trainings- und Testdaten mit Rauschen'
      }
    },
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        title: {
          display: true,
          text: 'x-Achse'
        }
      },
      y: {
        beginAtZero: true,
        min: -2,
        max: 3,
        title: {
          display: true,
          text: 'y-Funktionswerte'
        }
      }
    }
  }
});




async function print() {

  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({name: 'Model Summary'}, model);


  // Load and plot the original input data that we are going to train on.
  const values = jsonTrainingsdaten.map(d => ({
    x: d.x,
    y: d.y
  }));
  console.log("AUSGABE VALUES MY");
  console.log(values);

  tfvis.render.scatterplot(
    {name: 'Regression FFNN'},
    {values},
    {
      xLabel: 'x',
      yLabel: 'y',
      height: 300
    }
  );

  // More code will be added below

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(jsonTrainingsdaten);
  const {inputs, labels} = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log('Done Training');


}

document.addEventListener('DOMContentLoaded', print);




// Modell instanzieren
function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}



/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(jsonTrainingsdaten) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(jsonTrainingsdaten);

    // Step 2. Convert data to Tensor
    const inputs = jsonTrainingsdaten.map(d => d.x)
    const labels = jsonTrainingsdaten.map(d => d.y);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    //const inputMax = inputTensor.max();
    //const inputMin = inputTensor.min();
    //const labelMax = labelTensor.max();
    //const labelMin = labelTensor.min();

    //const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    //const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: inputs,
      labels: labels,
      // Return the min/max bounds so we can use them later.
      //inputMax,
      //inputMin,
      //labelMax,
      //labelMin,
    }
  });
}




async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 50;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}
