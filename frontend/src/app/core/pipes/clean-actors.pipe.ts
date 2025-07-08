import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
name: 'cleanActors'
})
export class CleanActorArrayPipe implements PipeTransform {
  transform(actors: string[]): string[] {
    if (!Array.isArray(actors)) return [];

    return actors.filter(actor => {
      const trimmed = actor.trim();
      return trimmed.length > 2 && !trimmed.startsWith('#');
    });
  }
}