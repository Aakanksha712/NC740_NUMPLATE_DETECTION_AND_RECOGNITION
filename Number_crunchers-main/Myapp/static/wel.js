/*var endpoint = '/api';

$.ajax({
	method: "GET",
	url: endpoint,
	success: function (data) {
		console.log(data);
	},
	error: function (error_data) {
		console.log(error_data);
	}
})*/

var endpoint = '/api';

$.ajax({
	method: "GET",
	url: endpoint,
	success: function (data) {
		console.log("Calling");
		drawLineGraph(data, 'myChartline');
		drawBarGraph(data, 'myChartBar');
		console.log("drawing");
	},
	error: function (error_data) {
		console.log(error_data);
	}
})


function drawLineGraph(data, id) {
	var labels = data.labels;
	var chartLabel = data.chartLabel;
	var chartdata = data.chartdata;
	var ctx = document.getElementById(id).getContext('2d');
	var chart = new Chart(ctx, {
		// The type of chart we want to create
		type: 'line',

		// The data for our dataset
		data: {
			labels: labels,
			datasets: [{
				label: chartLabel,
				backgroundColor: 'rgb(255, 100, 200)',
				borderColor: 'rgb(55, 99, 132)',
				data: chartdata,
			}]
		},

		// Configuration options go here
		options: {
			scales: {
				xAxes: [{
					display: true,
					title: "TimeStamp",
					scaleLabel: {
						display: true,
						labelString: 'TimeStamp'
					}
				}],
				yAxes: [{
					ticks: {
						beginAtZero: true,
						title: "Number_plate"
					},
					scaleLabel: {
						display: true,
						labelString: 'Number Plate No.'
					}
				}]
			}
		}

	});
}

function drawBarGraph(data, id) {
	var labels = data.labels;
	var chartLabel = data.chartLabel;
	var chartdata = data.chartdata;
	var ctx = document.getElementById(id).getContext('2d');
	var myChart = new Chart(ctx, {
		type: 'bar',
		data: {
			labels: labels,
			datasets: [{
				label: chartLabel,
				data: chartdata,
				backgroundColor: [
					'rgba(255, 99, 132, 0.2)',
					'rgba(54, 162, 235, 0.2)',
					'rgba(255, 206, 86, 0.2)',
					'rgba(75, 192, 192, 0.2)',
					'rgba(153, 102, 255, 0.2)',
					'rgba(255, 159, 64, 0.2)'
				],
				borderColor: [
					'rgba(255, 99, 132, 1)',
					'rgba(54, 162, 235, 1)',
					'rgba(255, 206, 86, 1)',
					'rgba(75, 192, 192, 1)',
					'rgba(153, 102, 255, 1)',
					'rgba(255, 159, 64, 1)'
				],
				borderWidth: 1
			}]
		},
		options: {
			scales: {
				yAxes: [{
					ticks: {
						beginAtZero: true
					}
				}]
			}
		}
	});
}

JSC.Chart('chart', {
	debug: true,
	legend_position: 'bottom right',
	type: 'area spline',
	defaultSeries: { shape_opacity: 0.5 },
	xAxis: {
		crosshair_enabled: true,
		scale: { type: 'time' }
	},
	title_label_text: 'Number of trucks',
	title_label_color: "orange",
	//yAxis: { formatString: 'c' }, 
	yAxis_label_color: "#FF0000",
	series: [
		{
			name: 'Truck entered',
			points: [
				['1/1/2020', 1005],
				['2/1/2020', 3202],
				['3/1/2020', 1345],
				['4/1/2020', 2341],
				['5/1/2020', 1440],
				['6/1/2020', 1780]
			]
		},
		{
			name: 'Truck Leave ',
			points: [
				['1/1/2020', 2698],
				['2/1/2020', 2795],
				['3/1/2020', 1954],
				['4/1/2020', 2702],
				['5/1/2020', 1230],
				['6/1/2020', 1110]
			]
		}
	]
});
