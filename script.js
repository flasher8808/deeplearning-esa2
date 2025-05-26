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

// Anzahl Trainingsdatensätze
const count = 100;

// Wertebereich der Trainingsdaten (x)
const min = -2;
const max = 2;

const xValues = generateRandomNumbers(count, min, max);
let yValues = [];
for (let i = 0; i < count; i++) {
  yValues.push(calcYValue(xValues[i]));
}

// Y-Werte berechnen
function calcYValue (xval) {
  //return xval*xval*xval;
  return 0.5*(xval+0.8)*(xval+1.8)*(xval-0.2)*(xval-0.3)*(xval-1.9)+1;
}



/*
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1)); // Wähle einen zufälligen Index aus 0 bis i
    // Tausche die Elemente an den Positionen i und j
    const temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
}


// Array mischen
shuffleArray(xValues);
*/


let trainingsdaten = xValues.slice(count/2);
let testdaten = xValues.slice(0, count/2);

let trainingsdatenY = [];
let testdatenY = [];


for (let i = 0; i < testdaten.length; i++) {
  testdatenY.push(calcYValue(testdaten[i]));
}

for (let i = 0; i < trainingsdaten.length; i++) {
  trainingsdatenY.push(calcYValue(trainingsdaten[i]));
}



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

















// JSON Objekte erzeugen
let jsonTrainingsdaten = trainingsdaten.map((value, index) => ({ x: value, y: trainingsdatenY[index]}));
let jsonTestdaten = testdaten.map((value, index) => ({ x: value, y: testdatenY[index]}));
let jsonTrainingsdatenVerrauscht = trainingsdaten.map((value, index) => ({ x: value, y: trainingsdatenY_rausch[index]}));
let jsonTestdatenVerrauscht = testdaten.map((value, index) => ({ x: value, y: testdatenY_rausch[index]}));

//console.log("Trainingsdaten");
//console.log(jsonTrainingsdaten);
//console.log("Testdaten");
//console.log(jsonTestdaten);
//console.log("Trainingsdaten verrauscht");
//console.log(trainingsdatenY_rausch);



//console.log("Testdaten Y ");
//console.log(testdatenY);
//console.log("Testdaten Y verrauscht");
//console.log(testdatenY_rausch);

//console.log("Trainingsdaten");
//console.log(trainingsdaten);
//console.log("Trainingsdaten Y");
//console.log(trainingsdatenY);

/*
// Fake Daten erzeugen
const fakeData = generateRandomNumbers(50, -2, 2);
let fakeDataY = [];

let jsonFakeData = fakeData.map((value, index) => ({ x: value, y: value*2}));
console.log(jsonFakeData);
*/




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

// R1 - Visualisierung Diagramm ohne Rauschen links
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


// R1 - Visualisierung Diagramm mit Rauschen rechts
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




// Aufruf R2
document.addEventListener('DOMContentLoaded', r2);


async function r2() {
  // 1. Trainingsdaten erzeugen
  /*
  const xs = [];
  const ys = [];
  for(let i = 0; i < 100; i++) {
    const x = -2 + 4 * i / 99;
    xs.push(x);
    ys.push(calcYValue(x));
  }
  */

  const xsTensor = tf.tensor2d(trainingsdaten, [trainingsdaten.length, 1]);
  const ysTensor = tf.tensor2d(trainingsdatenY, [trainingsdatenY.length, 1]);

  // 2. Modell definieren
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [1]}));
  model.add(tf.layers.dense({units: 100, activation: 'relu'}));
  model.add(tf.layers.dense({units: 1}));
  model.compile({optimizer: tf.train.adam(0.01), loss: 'meanSquaredError'});

  // 3. Training
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-text');

  /*
  await model.fit(xsTensor, ysTensor, {
    epochs: 100,
    batchSize: 32,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs, epochs) => {
        // Prozent berechnen
        const percent = ((epoch + 1) / epochs) * 100;
        progressBar.style.width = percent + '%';
        progressText.textContent = `Training läuft: Epoche ${epoch + 1} / 300 (${percent.toFixed(1)}%) — Loss: ${logs.loss.toFixed(5)}`;
        //console.log(`Epoch ${epoch + 1}: Verlust = ${logs.loss.toFixed(5)}`);
      },
      onTrainEnd: () => {
        progressText.textContent = 'Training abgeschlossen!';
        progressBar.style.backgroundColor = '#2196F3'; // Farbe ändern nach Ende, optional
        document.getElementById("progress-container").style.display = "none";
      }
    }
  });
  */

  // Anzahl Trainingsepochen
  const epochs = 100; 

  await model.fit(xsTensor, ysTensor, {
    epochs: epochs,
    batchSize: 32,
    shuffle: true,
    callbacks: createProgressCallback(epochs)
  });

  function createProgressCallback(totalEpochs) {
    return {
      onEpochEnd: (epoch, logs) => {
        const percent = ((epoch + 1) / totalEpochs) * 100;
        progressBar.style.width = percent + '%';
        progressText.textContent = `Training läuft: Epoche ${epoch + 1} / ${totalEpochs} (${percent.toFixed(1)}%) — Loss: ${logs.loss.toFixed(5)}`;
      },
      onTrainEnd: () => {
        progressText.textContent = 'Training abgeschlossen!';
        progressBar.style.backgroundColor = '#2196F3';
        document.getElementById("progress-container").style.display = "none";
      }
    }
  }



  /* FIRST WORKING
  await model.fit(xsTensor, ysTensor, {epochs: 300, batchSize: 32, shuffle: true, callbacks: {
    onEpochEnd: (epoch, logs) => {
      if(epoch % 50 === 0) {
        console.log(`Epoch ${epoch}: Loss = ${logs.loss.toFixed(5)}`);
      }
    }
  }});
  */

  // 4. Daten für Visualisierung vorbereiten

  // Werte Originalfunktion
  const originalPoints = trainingsdaten.map(x => ({x: x, y: calcYValue(x)}));

  // Werte Modellvorhersage
  // Damit nicht jeden Punkt asynchron abfragen (zu langsam), machen wir batch-wise Prediction
  const xTensorForPred = tf.tensor2d(trainingsdaten, [trainingsdaten.length, 1]);
  const yPredTensor = model.predict(xTensorForPred);
  const yPreds = await yPredTensor.array();
  xTensorForPred.dispose();
  yPredTensor.dispose();

  const predictedPoints = trainingsdaten.map((x, i) => ({x: x, y: yPreds[i][0]}));

  xsTensor.dispose();
  ysTensor.dispose();

  // 5. Chart.js Chart erstellen
  const r2ctxtrain = document.getElementById('r2-train').getContext('2d');
  const r2charttrain = new Chart(r2ctxtrain, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Modellvorhersage',
          data: predictedPoints,
          backgroundColor: 'rgba(75, 192, 192, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        {
          label: 'Trainingsdaten',
          data: originalPoints,
          backgroundColor: 'rgba(192, 75, 75, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          title: {
            display: true,
            text: 'x-Achse'
          },
          min: -2,
          max: 2
        },
        y: {
          title: {
            display: true,
            text: 'y-Funktionswerte'
          },
          min: -3,
          max: 3
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Vorhersage ohne Rauschen auf Trainingsdaten'
        }
      }
    }
  });

  // Werte Modellvorhersage Testdaten
  // Damit nicht jeden Punkt asynchron abfragen (zu langsam), machen wir batch-wise Prediction
  const xTensorForPredTest = tf.tensor2d(testdaten, [testdaten.length, 1]);
  const yPredTensorTest = model.predict(xTensorForPredTest);
  const yPredsTest = await yPredTensorTest.array();
  xTensorForPredTest.dispose();
  yPredTensorTest.dispose();

  const originalPointsTest = testdaten.map(x => ({x: x, y: calcYValue(x)}));
  const predictedPointsTest = testdaten.map((x, i) => ({x: x, y: yPredsTest[i][0]}));

  xsTensor.dispose();
  ysTensor.dispose();

  // 5. Chart.js Chart erstellen
  const r2ctxtest = document.getElementById('r2-test').getContext('2d');
  const r2charttest = new Chart(r2ctxtest, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Modellvorhersage',
          data: predictedPointsTest,
          backgroundColor: 'rgba(75, 192, 192, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        {
          label: 'Testdaten',
          data: originalPointsTest,
          backgroundColor: 'rgba(192, 75, 75, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          title: {
            display: true,
            text: 'x-Achse'
          },
          min: -2,
          max: 2
        },
        y: {
          title: {
            display: true,
            text: 'y-Funktionswerte'
          },
          min: -3,
          max: 3
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Vorhersage ohne Rauschen auf Testdaten'
        }
      }
    }
  });

  r3();
}






async function r3() {
  // 1. Trainingsdaten erzeugen
  /*
  const xs = [];
  const ys = [];
  for(let i = 0; i < 100; i++) {
    const x = -2 + 4 * i / 99;
    xs.push(x);
    ys.push(calcYValue(x));
  }
  */

  const xsTensor = tf.tensor2d(trainingsdaten, [trainingsdaten.length, 1]);
  const ysTensor = tf.tensor2d(trainingsdatenY_rausch, [trainingsdatenY_rausch.length, 1]);

  // 2. Modell definieren
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [1]}));
  model.add(tf.layers.dense({units: 100, activation: 'relu'}));
  model.add(tf.layers.dense({units: 1}));
  model.compile({optimizer: tf.train.adam(0.01), loss: 'meanSquaredError'});

  // 3. Training
  const progressBar = document.getElementById('r3-progress-bar');
  const progressText = document.getElementById('r3-progress-text');

  /*
  await model.fit(xsTensor, ysTensor, {
    epochs: 100,
    batchSize: 32,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs, epochs) => {
        // Prozent berechnen
        const percent = ((epoch + 1) / epochs) * 100;
        progressBar.style.width = percent + '%';
        progressText.textContent = `Training läuft: Epoche ${epoch + 1} / 300 (${percent.toFixed(1)}%) — Loss: ${logs.loss.toFixed(5)}`;
        //console.log(`Epoch ${epoch + 1}: Verlust = ${logs.loss.toFixed(5)}`);
      },
      onTrainEnd: () => {
        progressText.textContent = 'Training abgeschlossen!';
        progressBar.style.backgroundColor = '#2196F3'; // Farbe ändern nach Ende, optional
        document.getElementById("progress-container").style.display = "none";
      }
    }
  });
  */

  // Anzahl Trainingsepochen
  const epochs = 200; 

  await model.fit(xsTensor, ysTensor, {
    epochs: epochs,
    batchSize: 32,
    shuffle: true,
    callbacks: createProgressCallback(epochs)
  });

  function createProgressCallback(totalEpochs) {
    return {
      onEpochEnd: (epoch, logs) => {
        const percent = ((epoch + 1) / totalEpochs) * 100;
        progressBar.style.width = percent + '%';
        progressText.textContent = `Training läuft: Epoche ${epoch + 1} / ${totalEpochs} (${percent.toFixed(1)}%) — Loss: ${logs.loss.toFixed(5)}`;
      },
      onTrainEnd: () => {
        progressText.textContent = 'Training abgeschlossen!';
        progressBar.style.backgroundColor = '#2196F3';
        document.getElementById("r3-progress-container").style.display = "none";
      }
    }
  }


  // 4. Daten für Visualisierung vorbereiten

  // Werte Originalfunktion
  //trainingsdaten = trainingsdaten.sort(function(a, b){return a - b});
  const originalPoints = trainingsdaten.map((x, index) => ({x: x, y: trainingsdatenY_rausch[index]}));
  console.log(originalPoints);


  // Werte Modellvorhersage
  // Damit nicht jeden Punkt asynchron abfragen (zu langsam), machen wir batch-wise Prediction
  const xTensorForPred = tf.tensor2d(trainingsdaten, [trainingsdaten.length, 1]);
  const yPredTensor = model.predict(xTensorForPred);
  const yPreds = await yPredTensor.array();
  xTensorForPred.dispose();
  yPredTensor.dispose();

  const predictedPoints = trainingsdaten.map((x, i) => ({x: x, y: yPreds[i][0]}));

  xsTensor.dispose();
  ysTensor.dispose();

  // 5. Chart.js Chart erstellen
  const r3ctxtrain = document.getElementById('r3-train').getContext('2d');
  const r3charttrain = new Chart(r3ctxtrain, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Modellvorhersage',
          data: predictedPoints,
          backgroundColor: 'rgba(75, 192, 192, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        {
          label: 'Trainingsdaten',
          data: originalPoints,
          backgroundColor: 'rgba(192, 75, 75, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          title: {
            display: true,
            text: 'x-Achse'
          },
          min: -2,
          max: 2
        },
        y: {
          title: {
            display: true,
            text: 'y-Funktionswerte'
          },
          min: -3,
          max: 3
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Vorhersage mit Rauschen auf Trainingsdaten'
        }
      }
    }
  });

  // Werte Modellvorhersage Testdaten
  // Damit nicht jeden Punkt asynchron abfragen (zu langsam), machen wir batch-wise Prediction
  const xTensorForPredTest = tf.tensor2d(testdaten, [testdaten.length, 1]);
  const yPredTensorTest = model.predict(xTensorForPredTest);
  const yPredsTest = await yPredTensorTest.array();
  xTensorForPredTest.dispose();
  yPredTensorTest.dispose();

  const originalPointsTest = testdaten.map((x, index) => ({x: x, y: testdatenY_rausch[index]}));
  const predictedPointsTest = testdaten.map((x, i) => ({x: x, y: yPredsTest[i][0]}));

  xsTensor.dispose();
  ysTensor.dispose();

  // 5. Chart.js Chart erstellen
  const r3ctxtest = document.getElementById('r3-test').getContext('2d');
  const r3charttest = new Chart(r3ctxtest, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Modellvorhersage',
          data: predictedPointsTest,
          backgroundColor: 'rgba(75, 192, 192, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        {
          label: 'Testdaten',
          data: originalPointsTest,
          backgroundColor: 'rgba(192, 75, 75, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: {
          type: 'linear',
          position: 'bottom',
          title: {
            display: true,
            text: 'x-Achse'
          },
          min: -2,
          max: 2
        },
        y: {
          title: {
            display: true,
            text: 'y-Funktionswerte'
          },
          min: -3,
          max: 3
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Vorhersage mit Rauschen auf Testdaten'
        }
      }
    }
  });

}






















 



// R3

//document.addEventListener('DOMContentLoaded', r3print);



async function r3print() {

  // Create the model
  const r3model = r3createModel();
  tfvis.show.modelSummary({name: 'R3 Model Summary'}, r3model);


  // Load and plot the original input data that we are going to train on.
  const values = jsonTestdatenVerrauscht.map(d => ({
    x: d.x,
    y: d.y
  }));
  console.log("R3 AUSGABE VALUES MY");
  console.log(values);

  tfvis.render.scatterplot(
    {name: 'R3 Regression FFNN'},
    {values},
    {
      xLabel: 'x',
      yLabel: 'y',
      height: 300
    }
  );

  // More code will be added below

  // Convert the data to a form we can use for training.
  const r3tensorData = convertToTensor(jsonTrainingsdatenVerrauscht); //jsonTrainingsdaten
  const {inputs, labels} = r3tensorData;

  const r3tensorDataTest = convertToTensor(jsonTestdatenVerrauscht);
  // Train the model
  await r3trainModel(r3model, inputs, labels);
  console.log('R3 Done Training');

  // Make some predictions using the model and compare them to the
  // original data
  r3testModelTrain(r3model, jsonTrainingsdatenVerrauscht, r3tensorData);
  r3testModelTest(r3model, jsonTestdatenVerrauscht, r3tensorDataTest);

  

}

// Modell instanzieren
function r3createModel() {
  // Create a sequential model
  const r3model = tf.sequential();

  // Add a single input layer
  r3model.add(tf.layers.dense({inputShape: [1], units: 256, useBias: true}));

  // Add hidden middle layer
  r3model.add(tf.layers.dense({units: 256, activation: 'relu'}));

  // Add hidden middle layer
  r3model.add(tf.layers.dense({units: 128, activation: 'relu'}));

  // Add hidden middle layer
  r3model.add(tf.layers.dense({units: 64, activation: 'relu'}));

  // Add hidden middle layer
  r3model.add(tf.layers.dense({units: 32, activation: 'relu'}));

  // Add hidden middle layer
  r3model.add(tf.layers.dense({units: 16, activation: 'relu'}));

  // Add hidden middle layer
  r3model.add(tf.layers.dense({units: 8, activation: 'relu'}));
  

  // Add an output layer
  r3model.add(tf.layers.dense({units: 1, activation: 'linear'}));

  return r3model;
}


/* ORIGINAL
async function r3trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 8;
  const epochs = 50;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'R3 Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}
*/

async function r3trainModel(model, inputs, labels) {
  const learningRate = 0.001;

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 8;
  const epochs = 300;

  const updateProgressBar = (epoch) => {
    const progress = ((epoch + 1) / epochs) * 100;
    document.getElementById('r3-progress-bar').style.width = progress + '%';
    document.getElementById('r3-progress-text').innerText = 'Training zu ' + progress.toFixed(0) + '% abgeschlossen';
    if (progress == 100){
      document.getElementById('r3-progress-container').style.display = "none";
      document.getElementById('r3-train').style.display = "block";
      document.getElementById('r3-test').style.display = "block";
    }
    
  };

  const trainingCallbacks = tfvis.show.fitCallbacks(
    { name: 'R3 Training Performance' },
    ['loss', 'mse'],
    { height: 200, callbacks: ['onEpochEnd'] }
  );

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        updateProgressBar(epoch);
        await trainingCallbacks.onEpochEnd(epoch, logs);  // Sicherstellen, dass der tfvis Callback korrekt aufgerufen wird.
      }
    }
  });
}

function r3testModelTrain(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {

    const xsNorm = tf.linspace(0, 1, 100);
    const predictions = model.predict(xsNorm.reshape([100, 1]));

    const unNormXs = xsNorm
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = predictions
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });


  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPoints = inputData.map(d => ({
    x: d.x, y: d.y,
  }));

  //console.log("originalPoints");
  //console.log(originalPoints);
/*
  for (i = 0; i <= originalPoints.length; i++){
    console.log("x:");
    console.log(originalPoints[i]);
  }
  */

  const ctx_r3train = document.getElementById('r3-train').getContext('2d');
  const chartr3train = new Chart(ctx_r3train, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'original',
        data: originalPoints.map((value, index) => ({ x: value.x, y: originalPoints[index].y })),
        backgroundColor: 'rgba(192, 75, 75, 0.8)',
        pointRadius: 3
      },
      {
        label: 'predicted',
        data: predictedPoints.map((value, index) => ({ x: value.x, y: predictedPoints[index].y })),
        backgroundColor: 'rgba(75, 192, 192, 0.8)',
        pointRadius: 3
      }]
    },
    options: {
      plugins: {
        title: {
            display: true,
            text: 'Vorhersage mit Rauschen auf Trainingsdaten'
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
          min: -4, //-2,
          max:  4, //3,
          title: {
            display: true,
            text: 'y-Funktionswerte'
          }
        }
      }
    }
  });

  tfvis.render.scatterplot(
    {name: 'R3 Model Predictions vs Original Data Trainingdata'},
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: 'x',
      yLabel: 'y',
      height: 300
    }
  );
}


function r3testModelTest(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {

    const xsNorm = tf.linspace(0, 1, 100);
    const predictions = model.predict(xsNorm.reshape([100, 1]));

    const unNormXs = xsNorm
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = predictions
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });


  const predictedPointsTest = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPointsTest = inputData.map(d => ({
    x: d.x, y: d.y,
  }));

  const ctx_r3test = document.getElementById('r3-test').getContext('2d');
  const chartr3train = new Chart(ctx_r3test, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'original',
        data: originalPointsTest.map((value, index) => ({ x: value.x, y: originalPointsTest[index].y })),
        backgroundColor: 'rgba(192, 75, 75, 0.8)',
        pointRadius: 3
      },
      {
        label: 'predicted',
        data: predictedPointsTest.map((value, index) => ({ x: value.x, y: predictedPointsTest[index].y })),
        backgroundColor: 'rgba(75, 192, 192, 0.8)',
        pointRadius: 3
      }]
    },
    options: {
      plugins: {
        title: {
            display: true,
            text: 'Vorhersage mit Rauschen auf Testdaten'
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


  tfvis.render.scatterplot(
    {name: 'R3 Model Predictions vs Original Data Testdata'},
    {values: [originalPointsTest, predictedPointsTest], series: ['original', 'predicted']},
    {
      xLabel: 'x',
      yLabel: 'y',
      height: 300
    }
  );
}