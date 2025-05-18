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




















// Visualisierung Diagramm oben
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

// Visualisierung Diagramm ohne Rauschen links
const ctx_ohneRauschen = document.getElementById('ohneRauschen').getContext('2d');
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
const ctx_mitRauschen = document.getElementById('mitRauschen').getContext('2d');
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


