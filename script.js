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

  r4();
}




async function r4() {
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
  model.add(tf.layers.dense({units: 512, activation: 'relu', inputShape: [1]}));
  model.add(tf.layers.dense({units: 512, activation: 'relu'}));
  model.add(tf.layers.dense({units: 256, activation: 'relu'}));
  model.add(tf.layers.dense({units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: 1}));
  model.compile({optimizer: tf.train.adam(0.01), loss: 'meanSquaredError'});

  // 3. Training
  const progressBar = document.getElementById('r4-progress-bar');
  const progressText = document.getElementById('r4-progress-text');

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
  const epochs = 50; 

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
        document.getElementById("r4-progress-container").style.display = "none";
      }
    }
  }


  // 4. Daten für Visualisierung vorbereiten

  // Werte Originalfunktion
  //trainingsdaten = trainingsdaten.sort(function(a, b){return a - b});
  const originalPoints = trainingsdaten.map((x, index) => ({x: x, y: trainingsdatenY_rausch[index]}));
  //console.log(originalPoints);


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
  const r3ctxtrain = document.getElementById('r4-train').getContext('2d');
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
          text: 'Vorhersage mit Rauschen auf Trainingsdaten (overfit)'
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
  const r3ctxtest = document.getElementById('r4-test').getContext('2d');
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
          text: 'Vorhersage mit Rauschen auf Testdaten (overfit)'
        }
      }
    }
  });

}

















