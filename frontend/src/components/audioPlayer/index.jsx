import { useState, useEffect, useRef } from 'react';
import { ReactComponent as Play } from '../../assets/play.svg';
import { ReactComponent as Pause } from '../../assets/pause.svg';
import musicImg from '../../assets/music-file.png';
import './audioPlayer.css';

const AudioPlayer = ({ audioSrc, audioFile, fileRemove }) => {
	const [trackProgress, setTrackProgress] = useState(0);
	const [isPlaying, setIsPlaying] = useState(false);
	const [replay, setReplay] = useState(false);
	// console.log(audioSrc);

	const audioRef = useRef(new Audio(audioSrc));
	const intervalRef = useRef();
	const isReady = useRef(false);

	const { duration } = audioRef.current;
	// console.log(audioRef.current);
	// console.log(duration);

	const startTimer = () => {
		// Clear any timers already running
		clearInterval(intervalRef.current);

		intervalRef.current = setInterval(() => {
			if (audioRef.current.ended) {
				setIsPlaying(false);
			} else {
				setTrackProgress(audioRef.current.currentTime);
			}
		}, [10]);
	};

	// const onScrub = (value) => {
	// 	// Clear any timers already running
	// 	clearInterval(intervalRef.current);
	// 	audioRef.current.currentTime = value;
	// 	setTrackProgress(audioRef.current.currentTime);
	// };

	// const onScrubEnd = () => {
	// 	// If not already playing, start
	// 	if (!isPlaying) {
	// 		setIsPlaying(true);
	// 	}
	// 	startTimer();
	// };

	useEffect(() => {
		// console.log(isPlaying);
		if (isPlaying) {
			audioRef.current.play();
			startTimer();
		} else {
			audioRef.current.pause();
		}
	}, [isPlaying]);

	useEffect(() => {
		// console.log('hi');
		// Pause and clean up on unmount
		return () => {
			audioRef.current.pause();
			clearInterval(intervalRef.current);
		};
	}, []);

	// Handle setup when changing tracks
	useEffect(() => {
		// console.log('hello');
		audioRef.current.pause();

		audioRef.current = new Audio(audioSrc);
		setTrackProgress(audioRef.current.currentTime);

		if (isReady.current) {
			startTimer();
		} else {
			// Set the isReady ref as true for the next pass
			isReady.current = true;
		}
	}, [audioSrc, replay]);

	const currentPercentage = duration
		? `${(trackProgress / duration) * 100}%`
		: '0%';
	const trackStyling = `-webkit-gradient(linear, 0% 0%, 100% 0%, color-stop(${currentPercentage}, rgb(29, 78, 216)), color-stop(${currentPercentage}, #777))`;

	return (
		<div className='drop-file-preview__item'>
			<img src={musicImg} alt='Audio File' />
			<div className='drop-file-preview__item__info'>
				<p>{audioFile.name}</p>
				<p>{audioFile.size}B</p>
			</div>
			<span
				className='drop-file-preview__item__del'
				style={{ right: '60px' }}
			>
				{isPlaying ? (
					<button
						type='button'
						className='pause'
						onClick={() => setIsPlaying(false)}
						aria-label='Pause'
					>
						<Pause />
					</button>
				) : (
					<button
						type='button'
						className='play'
						onClick={() => setIsPlaying(true)}
						aria-label='Play'
					>
						<Play />
					</button>
				)}
			</span>
			<span
				className='drop-file-preview__item__del'
				onClick={() => fileRemove(audioFile)}
			>
				x
			</span>
			{/* <input
				type='range'
				value={trackProgress}
				step='0.01'
				min='0'
				max={duration ? duration : `${duration}`}
				className='progress'
				onChange={(e) => onScrub(e.target.value)}
				onMouseUp={onScrubEnd}
				onKeyUp={onScrubEnd}
				style={{ background: trackStyling }}
			/> */}
		</div>
	);
};

export default AudioPlayer;
