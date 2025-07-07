import { Component, OnInit, OnDestroy, ElementRef, ViewChild } from '@angular/core';
import { ApiCallService } from '../../../core/services/api/api-call.service';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { SubSink } from 'subsink';
import { NgxPaginationModule } from 'ngx-pagination';
import { FormBuilder, ReactiveFormsModule } from '@angular/forms';
import { Shared } from '../../../core/services/shared/shared';
import { ChangeDetectorRef } from '@angular/core';
import Chart from 'chart.js/auto';
import 'chartjs-adapter-date-fns';

@Component({
  selector: 'app-homepage',
  standalone: true,
  imports: [CommonModule, NgxPaginationModule, ReactiveFormsModule],
  templateUrl: './homepage.component.html',
  styleUrls: ['./homepage.component.scss'],
  providers: [ApiCallService]
})
export class HomepageComponent implements OnInit, OnDestroy {

  @ViewChild('chartCanvas', { static: false }) chartCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('timelineChart', { static: false }) timelineChart!: ElementRef<HTMLCanvasElement>;

  chart: Chart | undefined;
  timelineBarChart: Chart | undefined;

  subs = new SubSink();
  myForm: any;
  response: any = [];
  isGeneration: boolean = false;

  constructor(
    public apiService: ApiCallService,
    private http: HttpClient,
    private fb: FormBuilder,
    private sharedService: Shared,
    private cdr: ChangeDetectorRef
  ) {
    this.myForm = this.fb.group({
      message: ['']
    });
  }

  ngOnInit() {
    this.subs.sink = this.sharedService.result$.subscribe({
      next: (data: any) => {
        if (data) {
          this.result(data.text);
          this.cdr.markForCheck();
        }
      }
    });
  }

  result(text: string): void {
    const data = { q: text };
    this.isGeneration = true;
    this.response = [];

    this.subs.sink = this.apiService.generateResponse(data).subscribe({
      next: (res: any) => {
        this.response = res.semantic;
        this.isGeneration = false;

        // Pie Chart (Actors from Semantic[0])
        const actors = res.semantic[0]?.actors || [];
        const actorCount: { [key: string]: number } = {};
        actors.forEach((actor: any) => {
          actorCount[actor] = (actorCount[actor] || 0) + 1;
        });

        const labels = Object.keys(actorCount);
        const values = Object.values(actorCount);
        const backgroundColors = labels.map((_, i) => {
          const colors = [
            'rgb(255, 99, 132)',
            'rgb(54, 162, 235)',
            'rgb(255, 205, 86)',
            'rgb(75, 192, 192)',
            'rgb(153, 102, 255)',
            'rgb(255, 159, 64)',
            'rgb(201, 203, 207)'
          ];
          return colors[i % colors.length];
        });

        setTimeout(() => {
          const ctx = this.chartCanvas?.nativeElement.getContext('2d');
          if (ctx) {
            if (this.chart) this.chart.destroy();
            this.chart = new Chart(ctx, {
              type: 'pie',
              data: {
                labels: labels,
                datasets: [{
                  label: 'Actors in Semantic[0]',
                  data: values,
                  backgroundColor: backgroundColors,
                  hoverOffset: 4
                }]
              },
              options: {
                responsive: true,
                plugins: {
                  legend: { position: 'bottom' }
                }
              }
            });
          }
        }, 10);

        // Bar Chart (Timeline of Emails)
        const dateCount: { [key: string]: number } = {};
        res.semantic.forEach((item: any) => {
          if (item.date_sent) {
            const date = new Date(item.date_sent);
            const dateStr = date.toISOString().split('T')[0];
            dateCount[dateStr] = (dateCount[dateStr] || 0) + 1;
          }
        });

        const timelineLabels = Object.keys(dateCount).sort();
        const timelineValues = timelineLabels.map(date => dateCount[date]);
        console.log(timelineLabels, timelineValues, 'values')

        setTimeout(() => {
          const ctx = this.timelineChart?.nativeElement.getContext('2d');
          console.log(ctx, 'ctx')
          if (ctx) {
            if (this.timelineBarChart) this.timelineBarChart.destroy();
            this.timelineBarChart = new Chart(ctx, {
              type: 'bar',
              data: {
                labels: timelineLabels,
                datasets: [{
                  label: 'My First Dataset',
                  data: timelineValues,
                  backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(255, 159, 64, 0.2)',
                    'rgba(255, 205, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(201, 203, 207, 0.2)'
                  ],
                  borderColor: [
                    'rgb(255, 99, 132)',
                    'rgb(255, 159, 64)',
                    'rgb(255, 205, 86)',
                    'rgb(75, 192, 192)',
                    'rgb(54, 162, 235)',
                    'rgb(153, 102, 255)',
                    'rgb(201, 203, 207)'
                  ],
                  borderWidth: 1
                }]
              }
            });
          }
        }, 20);

        this.cdr.markForCheck();
      },
      error: (err: any) => {
        console.error(err);
        this.isGeneration = false;
        this.cdr.markForCheck();
      }
    });
  }

  submit(): void {
    const text = this.myForm.value.message;
    this.result(text);
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }
}
