//console.log('Hello TensorFlow');

// Anzahl Trainingsdatensätze
const count = 100;

// Wertebereich der Daten (x-Achse)
const min = -2;
const max = 2;



//let xValues = generateRandomNumbers(count, min, max);
let xValues = generateLinearSpace(count, min, max);
//shuffleArray(xValues);



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

// Gleichverteilte Zahlen erzeugen
function generateLinearSpace(numValues, min, max) {
  const arr = [];
  const step = (max - min) / (numValues - 1);
  for (let i = 0; i < numValues; i++) {
    arr.push(min + i * step);
  }
  console.log(arr);
  return arr;
}

// Array zufällig mischen
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

// Initiales Array aufteilen
function splitAlternating(array) {
  const arr1 = [];
  const arr2 = [];
  array.forEach((value, index) => {
    if (index % 2 === 0) {
      arr1.push(value);
    } else {
      arr2.push(value);
    }
  });
  return [arr1, arr2];
}


let [trainingsdaten, testdaten] = splitAlternating(xValues);






let yValues = [];
for (let i = 0; i < count; i++) {
  yValues.push(calcYValue(xValues[i]));
}

// Y-Werte berechnen
function calcYValue (xval) {
  //return xval*xval*xval;
  return 0.5*(xval+0.8)*(xval+1.8)*(xval-0.2)*(xval-0.3)*(xval-1.9)+1;
}




//let trainingsdaten = xValues.slice(count/2);
//let testdaten = xValues.slice(0, count/2);

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





// R1 - Visualisierung Diagramm ohne Rauschen links
const ctx_ohneRauschen = document.getElementById('r1-ohneRauschen').getContext('2d');
const ChartohneRauschen = new Chart(ctx_ohneRauschen, {
  type: 'scatter',
  data: {
    datasets: [{
      label: 'Trainingsdaten',
      data: trainingsdaten.map((value, index) => ({ x: value, y: calcYValue(value) })),
      backgroundColor: 'rgba(255, 136, 0, 0.8)',
      pointRadius: 3
    },
    {
      label: 'Testdaten',
      data: testdaten.map((value, index) => ({ x: value, y: calcYValue(value) })),
      backgroundColor: 'rgba(0, 119, 255, 0.8)',
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
        min: -3,
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
      backgroundColor: 'rgba(255, 136, 0, 0.8)',
      pointRadius: 3
    },
    {
      label: 'Testdaten',
      data: testdaten.map((value, index) => ({ x: value, y: testdatenY_rausch[index] })),
      backgroundColor: 'rgba(0, 119, 255, 0.8)',
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
        min: -3,
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
  let r2lossChart;
  initializeLossChart();

  const xsTensor = tf.tensor2d(trainingsdaten, [trainingsdaten.length, 1]);
  const ysTensor = tf.tensor2d(trainingsdatenY, [trainingsdatenY.length, 1]);
  // Tensoren für Loss
  const xsTestTensor = tf.tensor2d(testdaten, [testdaten.length, 1]);
  const ysTestTensor = tf.tensor2d(testdatenY, [testdatenY.length, 1]);

  // 2. Modell definieren
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [1], units: 256, activation: 'relu'}));
  model.add(tf.layers.dense({units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: 1}));
  model.compile({optimizer: tf.train.adam(0.01), loss: 'meanSquaredError'}); //metrics fehlen?

  // 3. Training
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-text');

  // Anzahl Trainingsepochen
  const epochs = 60; 

  // Diagramm für Trainings-Loss
  function initializeLossChart() {
    const ctx = document.getElementById('r2-loss-chart').getContext('2d');
    r2lossChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Loss (Trainingsdaten)',
            data: [],
            borderColor: 'rgba(0,119,255,0.8)',
            //backgroundColor: 'rgba(0,119,255,0.1)',
            fill: false,
            tension: 0.2,
            pointRadius: 0
          },
          {
            label: 'Loss (Testdaten)',
            data: [],
            borderColor: 'rgba(255,136,0,0.8)',
            backgroundColor: 'rgba(255,136,0,0.1)',
            fill: false,
            tension: 0.2,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        scales: {
          x: { title: { display: true, text: 'Epoche' } },
          y: { title: { display: true, text: 'Loss' } }
        },
        plugins: {
          title: {
            display: true,
            text: 'Loss-Verlauf für Trainings- und Testdaten'
          }
        }
      }
    });
  }



  await model.fit(xsTensor, ysTensor, {
    epochs: epochs,
    batchSize: 32,
    shuffle: true,
    validationData: [xsTestTensor, ysTestTensor],
    callbacks: createProgressCallback(epochs)
  });

  function createProgressCallback(totalEpochs) {
    return {
      onEpochEnd: (epoch, logs) => {
        const percent = ((epoch + 1) / totalEpochs) * 100;
        progressBar.style.width = percent + '%';
        progressText.textContent = `Training läuft: Epoche ${epoch + 1} / ${totalEpochs} (${percent.toFixed(1)}%) — Loss: ${logs.loss.toFixed(5)}`;
        document.getElementById("r2-train-loss").textContent = `Finaler Loss auf Trainingsdaten: ${logs.loss.toFixed(5)}`;
        //console.log(`Epoch ${epoch}: Loss = ${logs.loss.toFixed(5)}`);
        /*
        if((epoch + 20) % 20 === 0) {
          console.log(`Epoch ${epoch}: Loss = ${logs.loss.toFixed(5)}`);
        }*/
        if (r2lossChart) {
          r2lossChart.data.labels.push(epoch + 1);
          r2lossChart.data.datasets[0].data.push(logs.loss);
          r2lossChart.data.datasets[1].data.push(logs.val_loss);  // test/validation loss
          r2lossChart.update();
        }
      },
      onTrainEnd: () => {
        progressText.textContent = 'Training abgeschlossen!';
        progressBar.style.backgroundColor = '#2196F3';
        document.getElementById("progress-container").style.display = "none";
      }
    }
  }


  // Nach dem Training, falls du nochmal den Test Loss manuell berechnen willst:
  const predsTest = model.predict(xsTestTensor);
  const testLossTensor = tf.losses.meanSquaredError(ysTestTensor, predsTest);
  const testLossValue = (await testLossTensor.data())[0];
  //console.log('Finaler Test Loss:', testLossValue.toFixed(5));
  document.getElementById("r2-test-loss").textContent = `Finaler Loss auf Testdaten: ${testLossValue.toFixed(5)}`;


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
  xsTestTensor.dispose();
  ysTestTensor.dispose();
  predsTest.dispose();
  testLossTensor.dispose();


  // 5. Chart.js Chart erstellen
  const r2ctxtrain = document.getElementById('r2-train').getContext('2d');
  const r2charttrain = new Chart(r2ctxtrain, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Modellvorhersage',
          data: predictedPoints,
          backgroundColor: 'rgba(0, 119, 255, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        {
          label: 'Trainingsdaten',
          data: originalPoints,
          backgroundColor: 'rgba(255, 136, 0, 0.8)',
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


  // 5. Chart.js Chart erstellen
  const r2ctxtest = document.getElementById('r2-test').getContext('2d');
  const r2charttest = new Chart(r2ctxtest, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Modellvorhersage',
          data: predictedPointsTest,
          backgroundColor: 'rgba(0, 119, 255, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        {
          label: 'Testdaten',
          data: originalPointsTest,
          backgroundColor: 'rgba(255, 136, 0, 0.8)',
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


  // Loss Sichbarkeit umschalten
  document.getElementById("r2-train-loss").style.display = "block";
  document.getElementById("r2-test-loss").style.display = "block";


  // Aufruf R3
  r3();
}






async function r3() {

  let r3lossChart;
  initializeLossChart();

  const xsTensor = tf.tensor2d(trainingsdaten, [trainingsdaten.length, 1]);
  const ysTensor = tf.tensor2d(trainingsdatenY_rausch, [trainingsdatenY_rausch.length, 1]);
  // Tensoren für Loss
  const xsTestTensor = tf.tensor2d(testdaten, [testdaten.length, 1]);
  const ysTestTensor = tf.tensor2d(testdatenY, [testdatenY.length, 1]);


  // 2. Modell definieren
  const model = tf.sequential();
  const regularizer = tf.regularizers.l2({l2: 0.001});
  model.add(tf.layers.dense({inputShape: [1], units: 100, activation: 'relu', kernelRegularizer: regularizer}));
  model.add(tf.layers.dropout({rate: 0.3}));
  model.add(tf.layers.dense({units: 100, activation: 'relu', kernelRegularizer: regularizer}));
  model.add(tf.layers.dense({units: 1}));
  model.compile({optimizer: tf.train.adam(0.01), loss: 'meanSquaredError'}); //metrics fehlen?


  // 3. Training
  const progressBar = document.getElementById('r3-progress-bar');
  const progressText = document.getElementById('r3-progress-text');


  // Anzahl Trainingsepochen
  const epochs = 120;

  // Diagramm für Trainings-Loss
  function initializeLossChart() {
    const ctx = document.getElementById('r3-loss-chart').getContext('2d');
    r3lossChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Loss (Trainingsdaten)',
            data: [],
            borderColor: 'rgba(0,119,255,0.8)',
            //backgroundColor: 'rgba(0,119,255,0.1)',
            fill: false,
            tension: 0.2,
            pointRadius: 0
          },
          {
            label: 'Loss (Testdaten)',
            data: [],
            borderColor: 'rgba(255,136,0,0.8)',
            backgroundColor: 'rgba(255,136,0,0.1)',
            fill: false,
            tension: 0.2,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        scales: {
          x: { title: { display: true, text: 'Epoche' } },
          y: { title: { display: true, text: 'Loss' } }
        },
        plugins: {
          title: {
            display: true,
            text: 'Loss-Verlauf für Trainings- und Testdaten'
          }
        }
      }
    });
  }

  await model.fit(xsTensor, ysTensor, {
    epochs: epochs,
    batchSize: 32,
    shuffle: true,
    validationData: [xsTestTensor, ysTestTensor],
    callbacks: createProgressCallback(epochs)
  });

  function createProgressCallback(totalEpochs) {
    return {
      onEpochEnd: (epoch, logs) => {
        const percent = ((epoch + 1) / totalEpochs) * 100;
        progressBar.style.width = percent + '%';
        progressText.textContent = `Training läuft: Epoche ${epoch + 1} / ${totalEpochs} (${percent.toFixed(1)}%) — Loss: ${logs.loss.toFixed(5)}`;
        document.getElementById("r3-train-loss").textContent = `Finaler Loss auf Trainingsdaten: ${logs.loss.toFixed(5)}`;
        /*
        if((epoch + 20) % 20 === 0) {
          console.log(`Epoch ${epoch}: Loss = ${logs.loss.toFixed(5)}`);
        }*/
        if (r3lossChart) {
          r3lossChart.data.labels.push(epoch + 1);
          r3lossChart.data.datasets[0].data.push(logs.loss);
          r3lossChart.data.datasets[1].data.push(logs.val_loss);  // test/validation loss
          r3lossChart.update();
        }
      },
      onTrainEnd: () => {
        progressText.textContent = 'Training abgeschlossen!';
        progressBar.style.backgroundColor = '#2196F3';
        document.getElementById("r3-progress-container").style.display = "none";
      }
    }
  }


  // Nach dem Training, falls du nochmal den Test Loss manuell berechnen willst:
  const predsTest = model.predict(xsTestTensor);
  const testLossTensor = tf.losses.meanSquaredError(ysTestTensor, predsTest);
  const testLossValue = (await testLossTensor.data())[0];
  //console.log('Finaler Test Loss:', testLossValue.toFixed(5));
  document.getElementById("r3-test-loss").textContent = `Finaler Loss auf Testdaten: ${testLossValue.toFixed(5)}`;


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
          backgroundColor: 'rgba(0, 119, 255, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        {
          label: 'Trainingsdaten',
          data: originalPoints,
          backgroundColor: 'rgba(255, 136, 0, 0.8)',
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
          backgroundColor: 'rgba(0, 119, 255, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        {
          label: 'Testdaten',
          data: originalPointsTest,
          backgroundColor: 'rgba(255, 136, 0, 0.8)',
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

  // Loss Sichbarkeit umschalten
  document.getElementById("r3-train-loss").style.display = "block";
  document.getElementById("r3-test-loss").style.display = "block";

  r4();
}


function removeRandomItemsFromBoth(arr1, arr2, count) {
  if (!Array.isArray(arr1) || !Array.isArray(arr2)) {
    throw new Error("Beide Argumente müssen Arrays sein.");
  }
  if (arr1.length !== arr2.length) {
    throw new Error("Beide Arrays müssen die gleiche Länge haben.");
  }
  if (count < 0 || count > arr1.length) {
    throw new Error("Anzahl der zu löschenden Elemente ist ungültig.");
  }

  // Generiere zufällige eindeutige Indizes zum Entfernen
  const indicesToRemove = new Set();
  while (indicesToRemove.size < count) {
    const randomIndex = Math.floor(Math.random() * arr1.length);
    indicesToRemove.add(randomIndex);
  }

  // Filtere beide Arrays anhand der Zufallsindizes
  const filteredArr1 = arr1.filter((_, index) => !indicesToRemove.has(index));
  const filteredArr2 = arr2.filter((_, index) => !indicesToRemove.has(index));

  return [filteredArr1, filteredArr2];
}

async function r4() {
  
  // Trainingsdaten verkleinern

  const [newArr, newArrY] = removeRandomItemsFromBoth(trainingsdaten, trainingsdatenY_rausch, 35);
  console.log(newArr);

  let r4lossChart;
  initializeLossChart();

  const xsTensor = tf.tensor2d(newArr, [newArr.length, 1]);
  const ysTensor = tf.tensor2d(newArrY, [newArrY.length, 1]);
  // Tensoren für Loss
  const xsTestTensor = tf.tensor2d(testdaten, [testdaten.length, 1]);
  const ysTestTensor = tf.tensor2d(testdatenY, [testdatenY.length, 1]);

  // 2. Modell definieren
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [1], units: 800, activation: 'relu'}));
  model.add(tf.layers.dense({units: 800, activation: 'relu'}));
  //model.add(tf.layers.dense({units: 512, activation: 'relu'}));
  //model.add(tf.layers.dense({units: 512, activation: 'relu'}));
  model.add(tf.layers.dense({units: 1}));
  model.compile({optimizer: tf.train.adam(0.01), loss: 'meanSquaredError'}); //metrics fehlen?

  // 3. Training
  const progressBar = document.getElementById('r4-progress-bar');
  const progressText = document.getElementById('r4-progress-text');

  

  // Anzahl Trainingsepochen
  const epochs = 300; 

  // Diagramm für Trainings-Loss
  function initializeLossChart() {
    const ctx = document.getElementById('r4-loss-chart').getContext('2d');
    r4lossChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Loss (Trainingsdaten)',
            data: [],
            borderColor: 'rgba(0,119,255,0.8)',
            //backgroundColor: 'rgba(0,119,255,0.1)',
            fill: false,
            tension: 0.2,
            pointRadius: 0
          },
          {
            label: 'Loss (Testdaten)',
            data: [],
            borderColor: 'rgba(255,136,0,0.8)',
            backgroundColor: 'rgba(255,136,0,0.1)',
            fill: false,
            tension: 0.2,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        scales: {
          x: { title: { display: true, text: 'Epoche' } },
          y: { title: { display: true, text: 'Loss' } }
        },
        plugins: {
          title: {
            display: true,
            text: 'Loss-Verlauf für Trainings- und Testdaten'
          }
        }
      }
    });
  }

  await model.fit(xsTensor, ysTensor, {
    epochs: epochs,
    batchSize: 64,
    shuffle: true,
    validationData: [xsTestTensor, ysTestTensor],
    callbacks: createProgressCallback(epochs)
  });

  function createProgressCallback(totalEpochs) {
    return {
      onEpochEnd: (epoch, logs) => {
        const percent = ((epoch + 1) / totalEpochs) * 100;
        progressBar.style.width = percent + '%';
        progressText.textContent = `Training läuft: Epoche ${epoch + 1} / ${totalEpochs} (${percent.toFixed(1)}%) — Loss: ${logs.loss.toFixed(5)}`;
        document.getElementById("r4-train-loss").textContent = `Finaler Loss auf Trainingsdaten: ${logs.loss.toFixed(5)}`;
        if (r4lossChart) {
          r4lossChart.data.labels.push(epoch + 1);
          r4lossChart.data.datasets[0].data.push(logs.loss);
          r4lossChart.data.datasets[1].data.push(logs.val_loss);  // test/validation loss
          r4lossChart.update();
        }
      },
      onTrainEnd: () => {
        progressText.textContent = 'Training abgeschlossen!';
        progressBar.style.backgroundColor = '#2196F3';
        document.getElementById("r4-progress-container").style.display = "none";
      }
    }
  }


  // Nach dem Training, falls du nochmal den Test Loss manuell berechnen willst:
  const predsTest = model.predict(xsTestTensor);
  const testLossTensor = tf.losses.meanSquaredError(ysTestTensor, predsTest);
  const testLossValue = (await testLossTensor.data())[0];
  //console.log('Finaler Test Loss:', testLossValue.toFixed(5));
  document.getElementById("r4-test-loss").textContent = `Finaler Loss auf Testdaten: ${testLossValue.toFixed(5)}`;



  // 4. Daten für Visualisierung vorbereiten

  // Werte Originalfunktion
  //trainingsdaten = trainingsdaten.sort(function(a, b){return a - b});
  const originalPoints = newArr.map((x, index) => ({x: x, y: newArrY[index]}));
  //console.log(originalPoints);


  // Werte Modellvorhersage
  // Damit nicht jeden Punkt asynchron abfragen (zu langsam), machen wir batch-wise Prediction
  const xTensorForPred = tf.tensor2d(newArr, [newArr.length, 1]);
  const yPredTensor = model.predict(xTensorForPred);
  const yPreds = await yPredTensor.array();
  xTensorForPred.dispose();
  yPredTensor.dispose();

  const predictedPoints = newArr.map((x, i) => ({x: x, y: yPreds[i][0]}));

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
          backgroundColor: 'rgba(0, 119, 255, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        {
          label: 'Trainingsdaten',
          data: originalPoints,
          backgroundColor: 'rgba(255, 136, 0, 0.8)',
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
          backgroundColor: 'rgba(0, 119, 255, 0.8)',
          showLine: false,
          fill: false,
          pointRadius: 3,
          tension: 0.2
        },
        {
          label: 'Testdaten',
          data: originalPointsTest,
          backgroundColor: 'rgba(255, 136, 0, 0.8)',
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

  // Loss Sichbarkeit umschalten
  document.getElementById("r4-train-loss").style.display = "block";
  document.getElementById("r4-test-loss").style.display = "block";

}






// Bildervergrößerung
const thumbnails = document.querySelectorAll('.thumbnail');
const lightbox = document.getElementById('lightbox');
const lightboxImage = document.getElementById('lightboxImage');


document.querySelectorAll('.image-container').forEach(item => {
  item.addEventListener('click', event => {
      const imgSrc = item.querySelector('.image').src;
      
      lightboxImage.src = imgSrc;
      lightbox.style.display = 'flex'; 
    
  });
});


lightbox.addEventListener('click', function() {
    lightbox.style.display = 'none'; 
});


